"""
Comprehensive Benchmarking Framework for QMANN

Provides performance evaluation tools for comparing quantum-enhanced
neural networks against classical baselines using standard ML benchmarks.
"""

import time
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from ..core.base import QMANNBase
from ..core.exceptions import BenchmarkError


class Benchmarks:
    """Alias for PerformanceBenchmark for backward compatibility."""

    pass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    dataset_name: str
    task_type: str  # 'classification' or 'regression'

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    # Efficiency metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0

    # Quantum-specific metrics
    quantum_fidelity: Optional[float] = None
    quantum_memory_usage: Optional[float] = None
    circuit_depth: Optional[int] = None
    gate_count: Optional[int] = None

    # Additional metadata
    num_parameters: int = 0
    num_epochs: int = 0
    batch_size: int = 32
    learning_rate: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Model: {self.model_name}",
            f"Dataset: {self.dataset_name}",
            f"Task: {self.task_type}",
            "",
        ]

        if self.task_type == "classification":
            lines.extend(
                [
                    (
                        f"Accuracy: {self.accuracy:.4f}"
                        if self.accuracy
                        else "Accuracy: N/A"
                    ),
                    (
                        f"Precision: {self.precision:.4f}"
                        if self.precision
                        else "Precision: N/A"
                    ),
                    f"Recall: {self.recall:.4f}" if self.recall else "Recall: N/A",
                    (
                        f"F1-Score: {self.f1_score:.4f}"
                        if self.f1_score
                        else "F1-Score: N/A"
                    ),
                ]
            )
        elif self.task_type == "regression":
            lines.extend(
                [
                    f"MSE: {self.mse:.4f}" if self.mse else "MSE: N/A",
                    f"MAE: {self.mae:.4f}" if self.mae else "MAE: N/A",
                    f"R²: {self.r2_score:.4f}" if self.r2_score else "R²: N/A",
                ]
            )

        lines.extend(
            [
                "",
                f"Training Time: {self.training_time:.2f}s",
                f"Inference Time: {self.inference_time:.4f}s",
                f"Memory Usage: {self.memory_usage:.2f}MB",
                f"Parameters: {self.num_parameters:,}",
            ]
        )

        if self.quantum_fidelity is not None:
            lines.extend(
                [
                    "",
                    f"Quantum Fidelity: {self.quantum_fidelity:.4f}",
                    (
                        f"Circuit Depth: {self.circuit_depth}"
                        if self.circuit_depth
                        else "Circuit Depth: N/A"
                    ),
                    (
                        f"Gate Count: {self.gate_count}"
                        if self.gate_count
                        else "Gate Count: N/A"
                    ),
                ]
            )

        return "\n".join(lines)


class PerformanceBenchmark:
    """
    Performance benchmarking for QMANN models.

    Provides comprehensive evaluation against classical baselines
    using standard machine learning benchmarks.
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: List[BenchmarkResult] = []

        # Benchmark datasets
        self.datasets = {
            "synthetic_classification": self._generate_synthetic_classification,
            "synthetic_regression": self._generate_synthetic_regression,
            "continual_learning": self._generate_continual_learning_tasks,
            "memory_intensive": self._generate_memory_intensive_task,
        }

        # Classical baseline models
        self.baselines = {
            "lstm": self._create_lstm_baseline,
            "transformer": self._create_transformer_baseline,
            "mlp": self._create_mlp_baseline,
        }

    def run_comprehensive_benchmark(
        self,
        qmann_model,
        dataset_names: List[str] = None,
        baseline_names: List[str] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark comparing QMANN against baselines.

        Args:
            qmann_model: QMANN model to benchmark
            dataset_names: List of datasets to test on
            baseline_names: List of baseline models to compare against
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary of benchmark results by model type
        """
        if dataset_names is None:
            dataset_names = ["synthetic_classification", "synthetic_regression"]

        if baseline_names is None:
            baseline_names = ["lstm", "mlp"]

        all_results = {"qmann": [], "baselines": []}

        self.logger.info("Starting comprehensive benchmark...")

        for dataset_name in dataset_names:
            self.logger.info(f"Benchmarking on dataset: {dataset_name}")

            # Generate dataset
            train_loader, test_loader, task_info = self.datasets[dataset_name]()

            # Benchmark QMANN model
            qmann_result = self._benchmark_model(
                model=qmann_model,
                model_name="QMANN",
                train_loader=train_loader,
                test_loader=test_loader,
                task_info=task_info,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            all_results["qmann"].append(qmann_result)

            # Benchmark baseline models
            for baseline_name in baseline_names:
                baseline_model = self.baselines[baseline_name](task_info)
                baseline_result = self._benchmark_model(
                    model=baseline_model,
                    model_name=f"Baseline_{baseline_name}",
                    train_loader=train_loader,
                    test_loader=test_loader,
                    task_info=task_info,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                )
                all_results["baselines"].append(baseline_result)

        self.results.extend(all_results["qmann"])
        self.results.extend(all_results["baselines"])

        self.logger.info("Comprehensive benchmark completed")
        return all_results

    def _benchmark_model(
        self,
        model,
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        task_info: Dict[str, Any],
        num_epochs: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Benchmark a single model."""
        self.logger.info(f"Benchmarking model: {model_name}")

        # Initialize result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=task_info["name"],
            task_type=task_info["type"],
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_parameters=sum(p.numel() for p in model.parameters()),
        )

        # Setup training
        criterion = (
            nn.MSELoss() if task_info["type"] == "regression" else nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training phase
        start_time = time.time()
        memory_before = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_data, batch_targets in train_loader:
                optimizer.zero_grad()

                # Forward pass (handle different model interfaces)
                if (
                    hasattr(model, "forward")
                    and "use_quantum_memory" in model.forward.__code__.co_varnames
                ):
                    outputs, _, quantum_info = model(
                        batch_data, use_quantum_memory=True
                    )
                    if quantum_info and result.quantum_fidelity is None:
                        result.quantum_fidelity = quantum_info.get(
                            "quantum_fidelity", 0.0
                        )
                else:
                    outputs = model(batch_data)

                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        training_time = time.time() - start_time
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        result.training_time = training_time
        result.memory_usage = (memory_after - memory_before) / 1024 / 1024  # MB

        # Evaluation phase
        model.eval()
        all_predictions = []
        all_targets = []

        inference_start = time.time()
        with torch.no_grad():
            for batch_data, batch_targets in test_loader:
                if (
                    hasattr(model, "forward")
                    and "use_quantum_memory" in model.forward.__code__.co_varnames
                ):
                    outputs, _, _ = model(batch_data, use_quantum_memory=True)
                else:
                    outputs = model(batch_data)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())

        inference_time = time.time() - inference_start
        result.inference_time = inference_time / len(test_loader)

        # Compute metrics
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        if task_info["type"] == "classification":
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = targets.astype(int)

            result.accuracy = accuracy_score(true_classes, pred_classes)
            result.precision = precision_score(
                true_classes, pred_classes, average="weighted", zero_division=0
            )
            result.recall = recall_score(
                true_classes, pred_classes, average="weighted", zero_division=0
            )
            result.f1_score = f1_score(
                true_classes, pred_classes, average="weighted", zero_division=0
            )

        elif task_info["type"] == "regression":
            result.mse = mean_squared_error(targets, predictions)
            result.mae = mean_absolute_error(targets, predictions)
            result.r2_score = r2_score(targets, predictions)

        # Extract quantum-specific metrics if available
        if hasattr(model, "quantum_memory"):
            result.quantum_memory_usage = getattr(
                model.quantum_memory, "memory_usage", 0.0
            )
            if hasattr(model.quantum_memory, "encoding_circuit"):
                result.circuit_depth = model.quantum_memory.encoding_circuit.depth()
                result.gate_count = len(model.quantum_memory.encoding_circuit.data)

        self.logger.info(f"Completed benchmarking {model_name}")
        return result

    def _generate_synthetic_classification(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Generate synthetic classification dataset."""
        # Generate synthetic data
        n_samples = 1000
        n_features = 20
        n_classes = 3
        sequence_length = 10

        # Create sequential data
        X = torch.randn(n_samples, sequence_length, n_features)
        y = torch.randint(0, n_classes, (n_samples,))

        # Split into train/test
        split_idx = int(0.8 * n_samples)

        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        task_info = {
            "name": "synthetic_classification",
            "type": "classification",
            "input_size": n_features,
            "output_size": n_classes,
            "sequence_length": sequence_length,
        }

        return train_loader, test_loader, task_info

    def _generate_synthetic_regression(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Generate synthetic regression dataset."""
        # Generate synthetic data
        n_samples = 1000
        n_features = 15
        sequence_length = 8

        # Create sequential data with temporal dependencies
        X = torch.randn(n_samples, sequence_length, n_features)
        # Target is sum of last 3 timesteps with noise
        y = X[:, -3:, :5].sum(dim=(1, 2)) + 0.1 * torch.randn(n_samples)
        y = y.unsqueeze(1)  # Make it (n_samples, 1)

        # Split into train/test
        split_idx = int(0.8 * n_samples)

        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        task_info = {
            "name": "synthetic_regression",
            "type": "regression",
            "input_size": n_features,
            "output_size": 1,
            "sequence_length": sequence_length,
        }

        return train_loader, test_loader, task_info

    def _generate_continual_learning_tasks(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Generate continual learning benchmark."""
        # Create multiple tasks with different patterns
        n_samples_per_task = 200
        n_tasks = 3
        n_features = 10
        sequence_length = 5

        all_X = []
        all_y = []

        for task_id in range(n_tasks):
            # Each task has different data distribution
            X_task = (
                torch.randn(n_samples_per_task, sequence_length, n_features) + task_id
            )
            y_task = torch.full((n_samples_per_task,), task_id, dtype=torch.long)

            all_X.append(X_task)
            all_y.append(y_task)

        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)

        # Split into train/test
        split_idx = int(0.8 * len(X))

        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        task_info = {
            "name": "continual_learning",
            "type": "classification",
            "input_size": n_features,
            "output_size": n_tasks,
            "sequence_length": sequence_length,
            "num_tasks": n_tasks,
        }

        return train_loader, test_loader, task_info

    def _generate_memory_intensive_task(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Generate memory-intensive task for testing quantum memory advantages."""
        # Long sequences requiring memory
        n_samples = 500
        n_features = 8
        sequence_length = 50  # Long sequences

        X = torch.randn(n_samples, sequence_length, n_features)
        # Target depends on pattern from early in sequence
        y = (X[:, :5, :3].mean(dim=(1, 2)) > 0).float()

        # Split into train/test
        split_idx = int(0.8 * n_samples)

        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        test_dataset = TensorDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )  # Smaller batches
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        task_info = {
            "name": "memory_intensive",
            "type": "classification",
            "input_size": n_features,
            "output_size": 2,
            "sequence_length": sequence_length,
        }

        return train_loader, test_loader, task_info

    def _create_lstm_baseline(self, task_info: Dict[str, Any]) -> nn.Module:
        """Create LSTM baseline model."""

        class LSTMBaseline(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(self.dropout(lstm_out[:, -1, :]))  # Use last timestep
                return output

        return LSTMBaseline(
            input_size=task_info["input_size"],
            hidden_size=64,
            output_size=task_info["output_size"],
        )

    def _create_transformer_baseline(self, task_info: Dict[str, Any]) -> nn.Module:
        """Create Transformer baseline model."""

        class TransformerBaseline(nn.Module):
            def __init__(self, input_size, d_model, output_size, nhead=4, num_layers=2):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(d_model, output_size)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.input_projection(x)
                transformer_out = self.transformer(x)
                output = self.fc(
                    self.dropout(transformer_out[:, -1, :])
                )  # Use last timestep
                return output

        return TransformerBaseline(
            input_size=task_info["input_size"],
            d_model=64,
            output_size=task_info["output_size"],
        )

    def _create_mlp_baseline(self, task_info: Dict[str, Any]) -> nn.Module:
        """Create MLP baseline model."""

        class MLPBaseline(nn.Module):
            def __init__(self, input_size, sequence_length, output_size):
                super().__init__()
                flattened_size = input_size * sequence_length
                self.flatten = nn.Flatten()
                self.layers = nn.Sequential(
                    nn.Linear(flattened_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, output_size),
                )

            def forward(self, x):
                x = self.flatten(x)
                return self.layers(x)

        return MLPBaseline(
            input_size=task_info["input_size"],
            sequence_length=task_info["sequence_length"],
            output_size=task_info["output_size"],
        )

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        if not self.results:
            return "No benchmark results available."

        report_lines = ["QMANN Benchmark Comparison Report", "=" * 50, ""]

        # Group results by dataset
        datasets = {}
        for result in self.results:
            if result.dataset_name not in datasets:
                datasets[result.dataset_name] = []
            datasets[result.dataset_name].append(result)

        for dataset_name, dataset_results in datasets.items():
            report_lines.extend([f"Dataset: {dataset_name}", "-" * 30])

            # Find QMANN and baseline results
            qmann_results = [r for r in dataset_results if "QMANN" in r.model_name]
            baseline_results = [
                r for r in dataset_results if "Baseline" in r.model_name
            ]

            if qmann_results and baseline_results:
                qmann_result = qmann_results[0]

                # Performance comparison
                if qmann_result.task_type == "classification":
                    metric_name = "Accuracy"
                    qmann_metric = qmann_result.accuracy
                    baseline_metrics = [
                        r.accuracy for r in baseline_results if r.accuracy is not None
                    ]
                elif qmann_result.task_type == "regression":
                    metric_name = "R² Score"
                    qmann_metric = qmann_result.r2_score
                    baseline_metrics = [
                        r.r2_score for r in baseline_results if r.r2_score is not None
                    ]
                else:
                    continue

                if qmann_metric is not None and baseline_metrics:
                    best_baseline = max(baseline_metrics)
                    improvement = ((qmann_metric - best_baseline) / best_baseline) * 100

                    report_lines.extend(
                        [
                            f"QMANN {metric_name}: {qmann_metric:.4f}",
                            f"Best Baseline {metric_name}: {best_baseline:.4f}",
                            f"Improvement: {improvement:+.2f}%",
                            "",
                        ]
                    )

                # Efficiency comparison
                baseline_train_times = [r.training_time for r in baseline_results]
                if baseline_train_times:
                    avg_baseline_time = statistics.mean(baseline_train_times)
                    time_ratio = qmann_result.training_time / avg_baseline_time

                    report_lines.extend(
                        [
                            f"QMANN Training Time: {qmann_result.training_time:.2f}s",
                            f"Avg Baseline Training Time: {avg_baseline_time:.2f}s",
                            f"Time Ratio: {time_ratio:.2f}x",
                            "",
                        ]
                    )

                # Quantum-specific metrics
                if qmann_result.quantum_fidelity is not None:
                    report_lines.extend(
                        [
                            f"Quantum Fidelity: {qmann_result.quantum_fidelity:.4f}",
                            f"Circuit Depth: {qmann_result.circuit_depth}",
                            f"Gate Count: {qmann_result.gate_count}",
                            "",
                        ]
                    )

            report_lines.append("")

        return "\n".join(report_lines)

    def save_results(self, filepath: str) -> None:
        """Save benchmark results to file."""
        import json

        results_data = [result.to_dict() for result in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Benchmark results saved to {filepath}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all benchmarks."""
        if not self.results:
            return {}

        qmann_results = [r for r in self.results if "QMANN" in r.model_name]
        baseline_results = [r for r in self.results if "Baseline" in r.model_name]

        summary = {
            "total_benchmarks": len(self.results),
            "qmann_benchmarks": len(qmann_results),
            "baseline_benchmarks": len(baseline_results),
        }

        if qmann_results:
            summary["qmann_avg_training_time"] = statistics.mean(
                [r.training_time for r in qmann_results]
            )
            summary["qmann_avg_inference_time"] = statistics.mean(
                [r.inference_time for r in qmann_results]
            )

            # Quantum-specific averages
            quantum_fidelities = [
                r.quantum_fidelity
                for r in qmann_results
                if r.quantum_fidelity is not None
            ]
            if quantum_fidelities:
                summary["avg_quantum_fidelity"] = statistics.mean(quantum_fidelities)

        if baseline_results:
            summary["baseline_avg_training_time"] = statistics.mean(
                [r.training_time for r in baseline_results]
            )
            summary["baseline_avg_inference_time"] = statistics.mean(
                [r.inference_time for r in baseline_results]
            )

        return summary
