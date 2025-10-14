"""
Healthcare Predictive Analytics

QMANN application for healthcare prediction tasks including
patient outcome prediction, drug discovery, and personalized medicine.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..core.base import HybridComponent
from ..hybrid import QuantumLSTM, HybridTrainer
from ..core.exceptions import ApplicationError


class HealthcarePredictor(HybridComponent):
    """
    Healthcare prediction system using quantum memory-augmented neural networks.

    Designed for patient outcome prediction, treatment recommendation,
    and personalized medicine applications.
    """

    def __init__(
        self,
        config,
        input_features: int,
        prediction_horizon: int = 30,
        name: str = "HealthcarePredictor",
    ):
        super().__init__(config, name)

        self.input_features = input_features
        self.prediction_horizon = prediction_horizon

        # Core QMANN model
        self.qmann_model = QuantumLSTM(
            config=config,
            input_size=input_features,
            hidden_size=256,
            num_layers=3,
            quantum_memory_size=64,
            quantum_qubits=6,
            name=f"{name}_QMANN",
        )

        # Healthcare-specific layers
        self.risk_assessment = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.treatment_recommendation = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 treatment options
        )

        self.outcome_prediction = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, prediction_horizon),
        )

        # Patient memory for personalization
        self.patient_memory = {}

        # Training components
        self.trainer = HybridTrainer(self.qmann_model, config)

        self.logger.info(
            f"HealthcarePredictor initialized for {input_features} features"
        )

    def initialize(self) -> None:
        """Initialize the healthcare prediction system."""
        super().initialize()
        self.qmann_model.initialize()

        # Move to appropriate device
        device = self._get_device()
        self.to(device)

        self.logger.info("HealthcarePredictor initialization complete")

    def _get_device(self) -> torch.device:
        """Get appropriate device for computation."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def forward(self, patient_data: torch.Tensor, patient_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass through the healthcare predictor."""
        return self.predict_patient_outcome(patient_data, patient_id)

    def quantum_classical_interface(self, quantum_output: torch.Tensor, classical_input: torch.Tensor) -> torch.Tensor:
        """Interface between quantum and classical components."""
        # Combine quantum and classical outputs
        combined = torch.cat([quantum_output, classical_input], dim=-1)
        return combined

    def predict_patient_outcome(
        self,
        patient_data: torch.Tensor,
        patient_id: Optional[str] = None,
        include_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict patient outcomes using quantum-enhanced memory.

        Args:
            patient_data: Patient time series data (seq_len, features)
            patient_id: Optional patient identifier for personalization
            include_uncertainty: Whether to include uncertainty estimates

        Returns:
            Dictionary with predictions and metadata
        """
        if not self._initialized:
            self.initialize()

        self.qmann_model.eval()

        with torch.no_grad():
            # Add batch dimension
            if patient_data.dim() == 2:
                patient_data = patient_data.unsqueeze(0)

            # Get patient-specific memory if available
            hidden_state = self._get_patient_memory(patient_id)

            # Forward pass through QMANN
            qmann_output, new_hidden, quantum_info = self.qmann_model(
                patient_data, hidden_state=hidden_state, use_quantum_memory=True
            )

            # Store updated patient memory
            if patient_id:
                self._update_patient_memory(patient_id, new_hidden)

            # Extract final hidden state
            final_hidden = qmann_output[:, -1, :]  # Last time step

            # Generate predictions
            risk_score = self.risk_assessment(final_hidden)
            treatment_scores = self.treatment_recommendation(final_hidden)
            outcome_trajectory = self.outcome_prediction(final_hidden)

            predictions = {
                "risk_score": float(risk_score.squeeze().cpu()),
                "treatment_recommendations": torch.softmax(treatment_scores, dim=-1)
                .squeeze()
                .cpu()
                .numpy(),
                "outcome_trajectory": outcome_trajectory.squeeze().cpu().numpy(),
                "quantum_info": quantum_info,
            }

            # Add uncertainty estimates if requested
            if include_uncertainty:
                uncertainty_estimates = self._estimate_uncertainty(
                    patient_data, final_hidden, quantum_info
                )
                predictions["uncertainty"] = uncertainty_estimates

            return predictions

    def _get_patient_memory(
        self, patient_id: Optional[str]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve patient-specific memory state."""
        if patient_id and patient_id in self.patient_memory:
            return self.patient_memory[patient_id]
        return None

    def _update_patient_memory(
        self, patient_id: str, hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Update patient-specific memory state."""
        # Store only the most recent hidden state to avoid memory bloat
        self.patient_memory[patient_id] = (
            hidden_state[0].detach().clone(),
            hidden_state[1].detach().clone(),
        )

        # Limit memory size
        if len(self.patient_memory) > 1000:
            # Remove oldest entries
            oldest_key = next(iter(self.patient_memory))
            del self.patient_memory[oldest_key]

    def _estimate_uncertainty(
        self,
        patient_data: torch.Tensor,
        final_hidden: torch.Tensor,
        quantum_info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Estimate prediction uncertainty using quantum and classical methods."""
        # Quantum uncertainty from fidelity
        quantum_fidelity = quantum_info.get("quantum_fidelity", 0.0)
        quantum_uncertainty = 1.0 - quantum_fidelity

        # Classical uncertainty from model confidence
        # Use dropout at inference time for Monte Carlo estimation
        self.qmann_model.train()  # Enable dropout

        predictions = []
        for _ in range(10):  # Monte Carlo samples
            with torch.no_grad():
                output, _, _ = self.qmann_model(patient_data, use_quantum_memory=False)
                final_output = output[:, -1, :]
                risk = self.risk_assessment(final_output)
                predictions.append(float(risk.squeeze().cpu()))

        self.qmann_model.eval()  # Disable dropout

        classical_uncertainty = float(np.std(predictions))

        return {
            "quantum_uncertainty": quantum_uncertainty,
            "classical_uncertainty": classical_uncertainty,
            "combined_uncertainty": (quantum_uncertainty + classical_uncertainty) / 2,
            "prediction_variance": float(np.var(predictions)),
        }

    def train_on_patient_data(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the healthcare predictor on patient data.

        Args:
            train_data: Training data loader
            val_data: Optional validation data loader
            num_epochs: Number of training epochs

        Returns:
            Training results and statistics
        """
        if not self._initialized:
            self.initialize()

        # Define healthcare-specific loss function
        def healthcare_loss(outputs, targets):
            # Outputs: (risk, treatment, outcome)
            # Targets: (risk_target, treatment_target, outcome_target)

            risk_pred, treatment_pred, outcome_pred = outputs
            risk_target, treatment_target, outcome_target = targets

            # Risk prediction loss (binary cross-entropy)
            risk_loss = nn.BCELoss()(risk_pred, risk_target)

            # Treatment recommendation loss (cross-entropy)
            treatment_loss = nn.CrossEntropyLoss()(treatment_pred, treatment_target)

            # Outcome prediction loss (MSE)
            outcome_loss = nn.MSELoss()(outcome_pred, outcome_target)

            # Combined loss with weights
            total_loss = 0.4 * risk_loss + 0.3 * treatment_loss + 0.3 * outcome_loss

            return total_loss

        # Train the model
        training_results = self.trainer.train(
            train_loader=train_data,
            val_loader=val_data,
            num_epochs=num_epochs,
            loss_fn=healthcare_loss,
        )

        self.logger.info("Healthcare predictor training completed")
        return training_results

    def analyze_treatment_effectiveness(
        self,
        patient_data: torch.Tensor,
        treatment_history: List[int],
        outcome_history: List[float],
    ) -> Dict[str, Any]:
        """
        Analyze treatment effectiveness using quantum memory.

        Args:
            patient_data: Patient time series data
            treatment_history: List of treatment IDs
            outcome_history: List of outcome scores

        Returns:
            Treatment effectiveness analysis
        """
        if not self._initialized:
            self.initialize()

        # Store treatment-outcome pairs in quantum memory
        for treatment, outcome in zip(treatment_history, outcome_history):
            treatment_vector = np.zeros(10)
            treatment_vector[treatment] = outcome

            # Store in quantum memory for future retrieval
            self.qmann_model.quantum_memory.write(treatment_vector)

        # Analyze current patient for treatment recommendations
        predictions = self.predict_patient_outcome(patient_data)

        # Query quantum memory for similar cases
        current_state = patient_data[-1].cpu().numpy()  # Last observation
        similar_cases, similarities = self.qmann_model.quantum_memory.read(
            query=current_state, k=5, search_all_banks=True
        )

        # Analyze treatment patterns
        treatment_effectiveness = {}
        for case, similarity in zip(similar_cases, similarities):
            # Extract treatment and outcome from stored case
            bank_id, address, content = case
            treatment_id = np.argmax(content[:10])
            outcome_score = content[treatment_id]

            if treatment_id not in treatment_effectiveness:
                treatment_effectiveness[treatment_id] = []

            treatment_effectiveness[treatment_id].append(
                {
                    "outcome": outcome_score,
                    "similarity": similarity,
                    "confidence": similarity * outcome_score,
                }
            )

        # Summarize effectiveness
        effectiveness_summary = {}
        for treatment_id, cases in treatment_effectiveness.items():
            avg_outcome = np.mean([case["outcome"] for case in cases])
            avg_similarity = np.mean([case["similarity"] for case in cases])
            confidence = np.mean([case["confidence"] for case in cases])

            effectiveness_summary[treatment_id] = {
                "average_outcome": avg_outcome,
                "average_similarity": avg_similarity,
                "confidence_score": confidence,
                "num_similar_cases": len(cases),
            }

        return {
            "current_predictions": predictions,
            "treatment_effectiveness": effectiveness_summary,
            "similar_cases_found": len(similar_cases),
            "quantum_memory_stats": self.qmann_model.get_memory_statistics(),
        }

    def generate_personalized_treatment_plan(
        self,
        patient_data: torch.Tensor,
        patient_id: str,
        treatment_constraints: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Generate personalized treatment plan using quantum memory.

        Args:
            patient_data: Patient time series data
            patient_id: Patient identifier
            treatment_constraints: Optional list of allowed treatments

        Returns:
            Personalized treatment plan
        """
        # Get current predictions
        predictions = self.predict_patient_outcome(
            patient_data, patient_id=patient_id, include_uncertainty=True
        )

        # Get treatment recommendations
        treatment_scores = predictions["treatment_recommendations"]

        # Apply constraints if provided
        if treatment_constraints:
            constrained_scores = np.zeros_like(treatment_scores)
            for idx in treatment_constraints:
                if idx < len(treatment_scores):
                    constrained_scores[idx] = treatment_scores[idx]
            treatment_scores = constrained_scores

        # Rank treatments by score
        treatment_ranking = np.argsort(treatment_scores)[::-1]

        # Generate treatment plan
        treatment_plan = {
            "patient_id": patient_id,
            "risk_score": predictions["risk_score"],
            "recommended_treatments": [
                {
                    "treatment_id": int(treatment_id),
                    "confidence_score": float(treatment_scores[treatment_id]),
                    "rank": rank + 1,
                }
                for rank, treatment_id in enumerate(treatment_ranking[:5])
            ],
            "outcome_trajectory": predictions["outcome_trajectory"].tolist(),
            "uncertainty_estimates": predictions["uncertainty"],
            "quantum_enhancement": {
                "quantum_memory_used": predictions["quantum_info"][
                    "quantum_memory_used"
                ],
                "quantum_fidelity": predictions["quantum_info"]["quantum_fidelity"],
                "memory_hits": predictions["quantum_info"].get("memory_hits", 0),
            },
        }

        return treatment_plan

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        qmann_stats = self.qmann_model.get_memory_statistics()

        return {
            "qmann_statistics": qmann_stats,
            "patient_memory_size": len(self.patient_memory),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "model_components": {
                "qmann_model": str(self.qmann_model),
                "risk_assessment": str(self.risk_assessment),
                "treatment_recommendation": str(self.treatment_recommendation),
                "outcome_prediction": str(self.outcome_prediction),
            },
        }

    def clear_patient_memory(self, patient_id: Optional[str] = None) -> None:
        """Clear patient memory (specific patient or all)."""
        if patient_id:
            if patient_id in self.patient_memory:
                del self.patient_memory[patient_id]
                self.logger.debug(f"Cleared memory for patient {patient_id}")
        else:
            self.patient_memory.clear()
            self.logger.debug("Cleared all patient memory")

    def save_model(self, filepath: str) -> None:
        """Save the complete healthcare model."""
        model_data = {
            "qmann_state_dict": self.qmann_model.state_dict(),
            "risk_assessment_state_dict": self.risk_assessment.state_dict(),
            "treatment_recommendation_state_dict": self.treatment_recommendation.state_dict(),
            "outcome_prediction_state_dict": self.outcome_prediction.state_dict(),
            "patient_memory": self.patient_memory,
            "config": self.config.to_dict(),
            "model_statistics": self.get_model_statistics(),
        }

        torch.save(model_data, filepath)
        self.logger.info(f"Healthcare model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load the complete healthcare model."""
        model_data = torch.load(filepath, map_location=self._get_device())

        self.qmann_model.load_state_dict(model_data["qmann_state_dict"])
        self.risk_assessment.load_state_dict(model_data["risk_assessment_state_dict"])
        self.treatment_recommendation.load_state_dict(
            model_data["treatment_recommendation_state_dict"]
        )
        self.outcome_prediction.load_state_dict(
            model_data["outcome_prediction_state_dict"]
        )
        self.patient_memory = model_data.get("patient_memory", {})

        self.logger.info(f"Healthcare model loaded from {filepath}")
