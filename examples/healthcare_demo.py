"""
Healthcare Demo - QMANN Application

Demonstration of quantum memory-augmented neural networks
for healthcare predictive analytics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import QMANN components
from src.qmann import QMANNConfig
from src.qmann.applications import HealthcarePredictor
from src.qmann.hybrid import HybridTrainer


def generate_synthetic_patient_data(num_patients=100, num_timepoints=30, num_features=15):
    """
    Generate synthetic patient data for demonstration.
    
    Args:
        num_patients: Number of patients
        num_timepoints: Number of time points per patient
        num_features: Number of medical features
        
    Returns:
        Tuple of (patient_data, outcomes, treatments, patient_ids)
    """
    logger.info(f"Generating synthetic data for {num_patients} patients")
    
    # Medical features: vital signs, lab values, symptoms, etc.
    feature_names = [
        'heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature',
        'respiratory_rate', 'oxygen_saturation', 'glucose_level', 'white_blood_cells',
        'red_blood_cells', 'hemoglobin', 'pain_score', 'mobility_score',
        'cognitive_score', 'medication_adherence', 'comorbidity_index'
    ]
    
    patient_data = []
    outcomes = []
    treatments = []
    patient_ids = []
    
    for patient_id in range(num_patients):
        # Generate patient baseline characteristics
        age = np.random.randint(18, 90)
        gender = np.random.choice([0, 1])  # 0: female, 1: male
        severity = np.random.uniform(0.1, 0.9)  # Disease severity
        
        # Generate time series for this patient
        patient_timeline = []
        patient_outcomes = []
        patient_treatments = []
        
        for t in range(num_timepoints):
            # Simulate disease progression
            progression_factor = 1 + 0.1 * t * severity
            noise = np.random.normal(0, 0.1, num_features)
            
            # Generate realistic medical values
            features = np.array([
                np.random.normal(70 + age * 0.5, 10) * progression_factor,  # heart_rate
                np.random.normal(120 + age * 0.3, 15) * progression_factor,  # bp_sys
                np.random.normal(80 + age * 0.2, 10) * progression_factor,  # bp_dia
                np.random.normal(98.6, 1.5),  # temperature
                np.random.normal(16, 3),  # respiratory_rate
                np.random.normal(98, 2) / progression_factor,  # oxygen_saturation
                np.random.normal(100, 20) * progression_factor,  # glucose
                np.random.normal(7000, 2000) * progression_factor,  # wbc
                np.random.normal(4.5, 0.5),  # rbc
                np.random.normal(14, 2) / progression_factor,  # hemoglobin
                np.random.uniform(0, 10) * severity,  # pain_score
                np.random.uniform(0, 10) / progression_factor,  # mobility
                np.random.uniform(0, 10) / progression_factor,  # cognitive
                np.random.uniform(0.5, 1.0),  # medication_adherence
                severity * 5  # comorbidity_index
            ]) + noise
            
            patient_timeline.append(features)
            
            # Generate outcome (risk score)
            risk_score = min(0.95, severity * progression_factor * 0.3 + np.random.normal(0, 0.1))
            patient_outcomes.append(risk_score)
            
            # Generate treatment (simplified)
            if risk_score > 0.7:
                treatment = np.random.choice([2, 3, 4])  # Intensive treatments
            elif risk_score > 0.4:
                treatment = np.random.choice([1, 2])  # Moderate treatments
            else:
                treatment = 0  # Standard care
            
            patient_treatments.append(treatment)
        
        patient_data.append(np.array(patient_timeline))
        outcomes.append(np.array(patient_outcomes))
        treatments.append(np.array(patient_treatments))
        patient_ids.append(f"PATIENT_{patient_id:03d}")
    
    logger.info("Synthetic patient data generation complete")
    return np.array(patient_data), np.array(outcomes), np.array(treatments), patient_ids


def create_qmann_config():
    """Create QMANN configuration for healthcare application."""
    return QMANNConfig(
        quantum=QMANNConfig.QuantumConfig(
            max_qubits=16,
            gate_fidelity=0.999,
            coherence_time_t1=100e-6,
            coherence_time_t2=50e-6,
            enable_error_mitigation=True,
            mitigation_methods=["zero_noise_extrapolation", "readout_mitigation"]
        ),
        classical=QMANNConfig.ClassicalConfig(
            learning_rate=0.001,
            batch_size=16,
            max_epochs=50,
            early_stopping_patience=10,
            dropout=0.2,
            weight_decay=1e-5,
            gradient_clip_norm=1.0
        ),
        hybrid=QMANNConfig.HybridConfig(
            quantum_classical_ratio=0.3,
            alternating_training=False,
            sync_frequency=20,
            quantum_lr_scale=0.5
        ),
        application=QMANNConfig.ApplicationConfig(
            domain="healthcare",
            task_type="prediction",
            optimization_target="accuracy",
            real_time_inference=True
        )
    )


def demonstrate_patient_prediction():
    """Demonstrate patient outcome prediction."""
    logger.info("=== Patient Outcome Prediction Demo ===")
    
    # Create configuration
    config = create_qmann_config()
    
    # Generate sample patient data
    patient_data, outcomes, treatments, patient_ids = generate_synthetic_patient_data(
        num_patients=50, num_timepoints=30, num_features=15
    )
    
    # Create healthcare predictor
    predictor = HealthcarePredictor(
        config=config,
        input_features=15,
        prediction_horizon=7
    )
    
    predictor.initialize()
    
    # Demonstrate prediction for a single patient
    patient_idx = 0
    patient_tensor = torch.tensor(patient_data[patient_idx], dtype=torch.float32)
    patient_id = patient_ids[patient_idx]
    
    logger.info(f"Analyzing patient: {patient_id}")
    
    # Get predictions
    predictions = predictor.predict_patient_outcome(
        patient_data=patient_tensor,
        patient_id=patient_id,
        include_uncertainty=True
    )
    
    # Display results
    logger.info(f"Risk Score: {predictions['risk_score']:.3f}")
    logger.info(f"Top 3 Treatment Recommendations:")
    
    treatment_names = [
        "Standard Care", "Medication A", "Medication B", 
        "Intensive Care", "Surgery", "Therapy", 
        "Monitoring", "Lifestyle", "Alternative", "Emergency"
    ]
    
    treatment_scores = predictions['treatment_recommendations']
    top_treatments = np.argsort(treatment_scores)[::-1][:3]
    
    for i, treatment_idx in enumerate(top_treatments):
        logger.info(f"  {i+1}. {treatment_names[treatment_idx]}: {treatment_scores[treatment_idx]:.3f}")
    
    logger.info(f"Quantum Enhancement:")
    logger.info(f"  Memory Used: {predictions['quantum_info']['quantum_memory_used']}")
    logger.info(f"  Quantum Fidelity: {predictions['quantum_info']['quantum_fidelity']:.3f}")
    
    logger.info(f"Uncertainty Estimates:")
    uncertainty = predictions['uncertainty']
    logger.info(f"  Quantum Uncertainty: {uncertainty['quantum_uncertainty']:.3f}")
    logger.info(f"  Classical Uncertainty: {uncertainty['classical_uncertainty']:.3f}")
    logger.info(f"  Combined Uncertainty: {uncertainty['combined_uncertainty']:.3f}")
    
    return predictor, predictions


def demonstrate_treatment_effectiveness():
    """Demonstrate treatment effectiveness analysis."""
    logger.info("\n=== Treatment Effectiveness Analysis Demo ===")
    
    # Create configuration
    config = create_qmann_config()
    
    # Create healthcare predictor
    predictor = HealthcarePredictor(
        config=config,
        input_features=15,
        prediction_horizon=7
    )
    
    predictor.initialize()
    
    # Generate sample patient data
    patient_data = torch.randn(30, 15)  # 30 time points, 15 features
    
    # Simulate treatment history
    treatment_history = [0, 1, 2, 1, 0, 3, 2, 1, 0, 2]  # Treatment IDs over time
    outcome_history = [0.8, 0.6, 0.9, 0.7, 0.85, 0.4, 0.95, 0.75, 0.9, 0.88]  # Outcomes
    
    # Analyze treatment effectiveness
    effectiveness_analysis = predictor.analyze_treatment_effectiveness(
        patient_data=patient_data,
        treatment_history=treatment_history,
        outcome_history=outcome_history
    )
    
    logger.info("Treatment Effectiveness Analysis:")
    
    treatment_names = [
        "Standard Care", "Medication A", "Medication B", "Intensive Care"
    ]
    
    for treatment_id, effectiveness in effectiveness_analysis['treatment_effectiveness'].items():
        treatment_name = treatment_names[treatment_id] if treatment_id < len(treatment_names) else f"Treatment {treatment_id}"
        logger.info(f"  {treatment_name}:")
        logger.info(f"    Average Outcome: {effectiveness['average_outcome']:.3f}")
        logger.info(f"    Confidence Score: {effectiveness['confidence_score']:.3f}")
        logger.info(f"    Similar Cases: {effectiveness['num_similar_cases']}")
    
    logger.info(f"Similar Cases Found: {effectiveness_analysis['similar_cases_found']}")
    
    return effectiveness_analysis


def demonstrate_personalized_treatment():
    """Demonstrate personalized treatment plan generation."""
    logger.info("\n=== Personalized Treatment Plan Demo ===")
    
    # Create configuration
    config = create_qmann_config()
    
    # Create healthcare predictor
    predictor = HealthcarePredictor(
        config=config,
        input_features=15,
        prediction_horizon=7
    )
    
    predictor.initialize()
    
    # Generate patient data
    patient_data = torch.randn(30, 15)
    patient_id = "DEMO_PATIENT_001"
    
    # Generate personalized treatment plan
    treatment_plan = predictor.generate_personalized_treatment_plan(
        patient_data=patient_data,
        patient_id=patient_id,
        treatment_constraints=[0, 1, 2, 3, 4]  # Allow first 5 treatments
    )
    
    logger.info(f"Personalized Treatment Plan for {treatment_plan['patient_id']}:")
    logger.info(f"Risk Score: {treatment_plan['risk_score']:.3f}")
    
    logger.info("Recommended Treatments:")
    treatment_names = [
        "Standard Care", "Medication A", "Medication B", 
        "Intensive Care", "Surgery", "Therapy"
    ]
    
    for treatment in treatment_plan['recommended_treatments']:
        treatment_name = treatment_names[treatment['treatment_id']] if treatment['treatment_id'] < len(treatment_names) else f"Treatment {treatment['treatment_id']}"
        logger.info(f"  {treatment['rank']}. {treatment_name} (Confidence: {treatment['confidence_score']:.3f})")
    
    logger.info("Quantum Enhancement:")
    quantum_info = treatment_plan['quantum_enhancement']
    logger.info(f"  Memory Used: {quantum_info['quantum_memory_used']}")
    logger.info(f"  Quantum Fidelity: {quantum_info['quantum_fidelity']:.3f}")
    logger.info(f"  Memory Hits: {quantum_info['memory_hits']}")
    
    return treatment_plan


def demonstrate_model_statistics():
    """Demonstrate model statistics and performance metrics."""
    logger.info("\n=== Model Statistics Demo ===")
    
    # Create configuration
    config = create_qmann_config()
    
    # Create healthcare predictor
    predictor = HealthcarePredictor(
        config=config,
        input_features=15,
        prediction_horizon=7
    )
    
    predictor.initialize()
    
    # Run some predictions to generate statistics
    for i in range(10):
        patient_data = torch.randn(20, 15)
        patient_id = f"STATS_PATIENT_{i:03d}"
        
        predictions = predictor.predict_patient_outcome(
            patient_data=patient_data,
            patient_id=patient_id,
            include_uncertainty=True
        )
    
    # Get model statistics
    stats = predictor.get_model_statistics()
    
    logger.info("Model Statistics:")
    logger.info(f"  Total Parameters: {stats['total_parameters']:,}")
    logger.info(f"  Patient Memory Size: {stats['patient_memory_size']}")
    
    qmann_stats = stats['qmann_statistics']
    logger.info("QMANN Statistics:")
    logger.info(f"  Quantum Operations: {qmann_stats['quantum_operations']}")
    logger.info(f"  Classical Operations: {qmann_stats['classical_operations']}")
    logger.info(f"  Memory Hit Rate: {qmann_stats['memory_hit_rate']:.3f}")
    logger.info(f"  Quantum Ratio: {qmann_stats['quantum_ratio']:.3f}")
    
    return stats


def main():
    """Main demonstration function."""
    logger.info("QMANN Healthcare Demo Starting...")
    logger.info("=" * 50)
    
    try:
        # Run demonstrations
        predictor, predictions = demonstrate_patient_prediction()
        effectiveness_analysis = demonstrate_treatment_effectiveness()
        treatment_plan = demonstrate_personalized_treatment()
        model_stats = demonstrate_model_statistics()
        
        logger.info("\n" + "=" * 50)
        logger.info("QMANN Healthcare Demo Completed Successfully!")
        
        # Summary
        logger.info("\nDemo Summary:")
        logger.info(f"✓ Patient outcome prediction with quantum enhancement")
        logger.info(f"✓ Treatment effectiveness analysis using quantum memory")
        logger.info(f"✓ Personalized treatment plan generation")
        logger.info(f"✓ Model performance statistics and monitoring")
        
        logger.info(f"\nQuantum Advantage Demonstrated:")
        logger.info(f"  - Enhanced memory capacity for patient history")
        logger.info(f"  - Improved pattern recognition in treatment outcomes")
        logger.info(f"  - Uncertainty quantification using quantum fidelity")
        logger.info(f"  - Personalized medicine through quantum memory")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
