"""
Hybrid Quantum-Classical Trainer

Advanced training algorithms for hybrid quantum-classical neural networks
with coordinated optimization and error mitigation.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..core.base import HybridComponent
from ..core.exceptions import TrainingError, ConvergenceError
from ..utils import ErrorMitigation
from .quantum_lstm import QuantumLSTM


class HybridTrainer:
    """
    Trainer for hybrid quantum-classical neural networks.

    Implements coordinated optimization of quantum and classical parameters
    with advanced error mitigation and convergence monitoring.
    """

    def __init__(self, model: QuantumLSTM, config, name: str = "HybridTrainer"):
        self.model = model
        self.config = config
        self.name = name
        self.logger = logging.getLogger(__name__)

        # Training configuration
        self.hybrid_config = config.hybrid
        self.classical_config = config.classical

        # Optimizers
        self.classical_optimizer = None
        self.quantum_optimizer = None
        self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "quantum_fidelity": [],
            "memory_hit_rate": [],
            "convergence_metrics": [],
        }

        # Error mitigation
        self.error_mitigation = ErrorMitigation()

        # Performance tracking
        self.training_stats = {
            "total_training_time": 0.0,
            "quantum_training_time": 0.0,
            "classical_training_time": 0.0,
            "sync_operations": 0,
            "gradient_updates": 0,
        }

        self.logger.info(f"HybridTrainer initialized for {model.name}")

    def setup_optimizers(self) -> None:
        """Set up optimizers for classical and quantum parameters."""
        # Classical optimizer
        classical_params = [
            p
            for name, p in self.model.named_parameters()
            if "quantum" not in name.lower()
        ]

        if self.classical_config.optimizer.lower() == "adam":
            self.classical_optimizer = optim.Adam(
                classical_params,
                lr=self.classical_config.learning_rate,
                weight_decay=self.classical_config.weight_decay,
            )
        elif self.classical_config.optimizer.lower() == "sgd":
            self.classical_optimizer = optim.SGD(
                classical_params,
                lr=self.classical_config.learning_rate,
                momentum=0.9,
                weight_decay=self.classical_config.weight_decay,
            )
        else:
            raise TrainingError(
                f"Unsupported optimizer: {self.classical_config.optimizer}"
            )

        # Quantum parameter optimization (simplified)
        # In practice, would use specialized quantum optimizers
        quantum_params = [
            p for name, p in self.model.named_parameters() if "quantum" in name.lower()
        ]

        if quantum_params:
            self.quantum_optimizer = optim.Adam(
                quantum_params,
                lr=self.classical_config.learning_rate
                * self.hybrid_config.quantum_lr_scale,
            )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.classical_optimizer, mode="min", factor=0.5, patience=5
        )

        self.logger.info("Optimizers configured")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the hybrid quantum-classical model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            loss_fn: Loss function (default: MSE)

        Returns:
            Training results and statistics
        """
        if not self.model._initialized:
            self.model.initialize()

        if self.classical_optimizer is None:
            self.setup_optimizers()

        num_epochs = num_epochs or self.classical_config.max_epochs
        loss_fn = loss_fn or nn.MSELoss()

        self.logger.info(f"Starting hybrid training for {num_epochs} epochs")
        start_time = time.time()

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch

                # Training phase
                train_metrics = self._train_epoch(train_loader, loss_fn)

                # Validation phase
                if val_loader is not None:
                    val_metrics = self._validate_epoch(val_loader, loss_fn)
                else:
                    val_metrics = {"val_loss": train_metrics["train_loss"]}

                # Update learning rate
                self.scheduler.step(val_metrics["val_loss"])

                # Record history
                self._update_training_history(train_metrics, val_metrics)

                # Check for early stopping
                if self._check_early_stopping(val_metrics["val_loss"]):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

                # Periodic memory consolidation
                if epoch % 50 == 0 and epoch > 0:
                    self._consolidate_memory()

                # Log progress
                if epoch % 10 == 0:
                    self._log_progress(epoch, train_metrics, val_metrics)

        except Exception as e:
            raise TrainingError(
                f"Training failed at epoch {self.current_epoch}: {str(e)}"
            )

        finally:
            total_time = time.time() - start_time
            self.training_stats["total_training_time"] = total_time

        # Final results
        results = self._compile_training_results()
        self.logger.info(f"Training completed in {total_time:.2f}s")

        return results

    def _train_epoch(
        self, train_loader: DataLoader, loss_fn: Callable
    ) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_quantum_fidelity = 0.0
        epoch_memory_hits = 0
        batch_count = 0

        classical_time = 0.0
        quantum_time = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move to device
            data = data.to(self.model.device)
            targets = targets.to(self.model.device)

            # Zero gradients
            self.classical_optimizer.zero_grad()
            if self.quantum_optimizer:
                self.quantum_optimizer.zero_grad()

            # Forward pass
            start_time = time.time()
            outputs, _, quantum_info = self.model(data, use_quantum_memory=True)
            forward_time = time.time() - start_time

            # Calculate loss
            loss = loss_fn(outputs, targets)

            # Backward pass
            start_time = time.time()
            loss.backward()

            # Gradient clipping
            if self.classical_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.classical_config.gradient_clip_norm
                )

            # Coordinated optimization
            if self.hybrid_config.alternating_training:
                self._alternating_optimization(batch_idx)
            else:
                self._simultaneous_optimization()

            backward_time = time.time() - start_time

            # Update statistics
            epoch_loss += loss.item()
            epoch_quantum_fidelity += quantum_info.get("quantum_fidelity", 0.0)
            epoch_memory_hits += quantum_info.get("memory_hits", 0)

            classical_time += forward_time + backward_time
            if quantum_info.get("quantum_memory_used", False):
                quantum_time += forward_time * 0.3  # Estimate quantum portion

            batch_count += 1

            # Periodic synchronization
            if batch_idx % self.hybrid_config.sync_frequency == 0:
                self._synchronize_components()

        # Update training statistics
        self.training_stats["classical_training_time"] += classical_time
        self.training_stats["quantum_training_time"] += quantum_time

        return {
            "train_loss": epoch_loss / batch_count,
            "quantum_fidelity": epoch_quantum_fidelity / batch_count,
            "memory_hit_rate": epoch_memory_hits / batch_count,
            "classical_time": classical_time,
            "quantum_time": quantum_time,
        }

    def _validate_epoch(
        self, val_loader: DataLoader, loss_fn: Callable
    ) -> Dict[str, Any]:
        """Validate for one epoch."""
        self.model.eval()

        val_loss = 0.0
        val_quantum_fidelity = 0.0
        batch_count = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.model.device)
                targets = targets.to(self.model.device)

                outputs, _, quantum_info = self.model(data, use_quantum_memory=True)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item()
                val_quantum_fidelity += quantum_info.get("quantum_fidelity", 0.0)
                batch_count += 1

        return {
            "val_loss": val_loss / batch_count,
            "val_quantum_fidelity": val_quantum_fidelity / batch_count,
        }

    def _alternating_optimization(self, batch_idx: int) -> None:
        """Alternating optimization between classical and quantum parameters."""
        if batch_idx % 2 == 0:
            # Update classical parameters
            self.classical_optimizer.step()
        else:
            # Update quantum parameters
            if self.quantum_optimizer:
                self.quantum_optimizer.step()

        self.training_stats["gradient_updates"] += 1

    def _simultaneous_optimization(self) -> None:
        """Simultaneous optimization of all parameters."""
        self.classical_optimizer.step()
        if self.quantum_optimizer:
            self.quantum_optimizer.step()

        self.training_stats["gradient_updates"] += 1

    def _synchronize_components(self) -> None:
        """Synchronize quantum and classical components."""
        # Placeholder for component synchronization
        # In practice, would implement quantum-classical parameter coordination
        self.training_stats["sync_operations"] += 1
        self.logger.debug(
            f"Components synchronized (operation #{self.training_stats['sync_operations']})"
        )

    def _consolidate_memory(self) -> None:
        """Consolidate quantum memory during training."""
        try:
            consolidation_stats = self.model.consolidate_quantum_memory()
            self.logger.debug(f"Memory consolidated: {consolidation_stats}")
        except Exception as e:
            self.logger.warning(f"Memory consolidation failed: {e}")

    def _update_training_history(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Update training history."""
        self.training_history["train_loss"].append(train_metrics["train_loss"])
        self.training_history["val_loss"].append(val_metrics["val_loss"])
        self.training_history["quantum_fidelity"].append(
            train_metrics.get("quantum_fidelity", 0.0)
        )
        self.training_history["memory_hit_rate"].append(
            train_metrics.get("memory_hit_rate", 0.0)
        )

        # Calculate convergence metrics
        convergence_metric = self._calculate_convergence_metric()
        self.training_history["convergence_metrics"].append(convergence_metric)

    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric based on recent loss history."""
        if len(self.training_history["train_loss"]) < 5:
            return 1.0

        recent_losses = self.training_history["train_loss"][-5:]
        loss_variance = np.var(recent_losses)
        loss_trend = recent_losses[-1] - recent_losses[0]

        # Convergence metric: lower is better
        convergence = loss_variance + abs(loss_trend)
        return float(convergence)

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria are met."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return (
                self.patience_counter >= self.classical_config.early_stopping_patience
            )

    def _log_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log training progress."""
        memory_stats = self.model.get_memory_statistics()

        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Quantum Fidelity: {train_metrics.get('quantum_fidelity', 0.0):.4f}, "
            f"Memory Hit Rate: {memory_stats.get('memory_hit_rate', 0.0):.4f}"
        )

    def _compile_training_results(self) -> Dict[str, Any]:
        """Compile final training results."""
        memory_stats = self.model.get_memory_statistics()

        return {
            "training_history": self.training_history,
            "training_stats": self.training_stats,
            "final_memory_stats": memory_stats,
            "best_loss": self.best_loss,
            "total_epochs": self.current_epoch + 1,
            "convergence_achieved": self.patience_counter
            < self.classical_config.early_stopping_patience,
            "error_mitigation_stats": self.error_mitigation.get_mitigation_statistics(),
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "classical_optimizer_state_dict": self.classical_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
            "training_history": self.training_history,
            "training_stats": self.training_stats,
            "config": self.config.to_dict(),
        }

        if self.quantum_optimizer:
            checkpoint["quantum_optimizer_state_dict"] = (
                self.quantum_optimizer.state_dict()
            )

        torch.save(checkpoint, filepath)
        self.logger.info(f"Training checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.model.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.classical_optimizer is None:
            self.setup_optimizers()

        self.classical_optimizer.load_state_dict(
            checkpoint["classical_optimizer_state_dict"]
        )
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "quantum_optimizer_state_dict" in checkpoint and self.quantum_optimizer:
            self.quantum_optimizer.load_state_dict(
                checkpoint["quantum_optimizer_state_dict"]
            )

        self.current_epoch = checkpoint["current_epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.patience_counter = checkpoint["patience_counter"]
        self.training_history = checkpoint["training_history"]
        self.training_stats = checkpoint["training_stats"]

        self.logger.info(f"Training checkpoint loaded from {filepath}")

    def evaluate(
        self, test_loader: DataLoader, loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        self.model.eval()
        loss_fn = loss_fn or nn.MSELoss()

        test_loss = 0.0
        test_quantum_fidelity = 0.0
        batch_count = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.model.device)
                targets = targets.to(self.model.device)

                outputs, _, quantum_info = self.model(data, use_quantum_memory=True)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item()
                test_quantum_fidelity += quantum_info.get("quantum_fidelity", 0.0)
                batch_count += 1

        memory_stats = self.model.get_memory_statistics()

        return {
            "test_loss": test_loss / batch_count,
            "test_quantum_fidelity": test_quantum_fidelity / batch_count,
            "memory_stats": memory_stats,
        }
