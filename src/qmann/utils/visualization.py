"""
Visualization Tools for QMANN Analysis

Provides comprehensive visualization capabilities for quantum-classical
hybrid neural networks, including performance plots, quantum state
visualizations, and training dynamics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.exceptions import VisualizationError


class Visualization:
    """Alias for QMANNVisualizer for backward compatibility."""
    pass


class QMANNVisualizer:
    """
    Comprehensive visualization toolkit for QMANN analysis.
    
    Provides plotting capabilities for performance metrics, quantum states,
    training dynamics, and comparative analysis.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use(style)
        self.figsize = figsize
        
        # Color schemes
        self.colors = {
            "qmann": "#2E86AB",
            "classical": "#A23B72", 
            "quantum": "#F18F01",
            "hybrid": "#C73E1D",
            "baseline": "#7209B7"
        }
        
        # Initialize plotting parameters
        sns.set_palette("husl")
        
    def plot_training_dynamics(
        self,
        training_history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show_quantum_metrics: bool = True
    ) -> None:
        """
        Plot training dynamics including loss, accuracy, and quantum metrics.
        
        Args:
            training_history: Dictionary containing training metrics over epochs
            save_path: Optional path to save the plot
            show_quantum_metrics: Whether to include quantum-specific metrics
        """
        try:
            # Determine subplot layout
            n_plots = 2  # Loss and main metric
            if show_quantum_metrics and "quantum_fidelity" in training_history:
                n_plots += 1
            
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
            if n_plots == 1:
                axes = [axes]
            
            epochs = range(1, len(training_history["loss"]) + 1)
            
            # Plot loss
            axes[0].plot(epochs, training_history["loss"], 
                        color=self.colors["qmann"], linewidth=2, label="Training Loss")
            axes[0].set_title("Training Loss", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot main performance metric
            if "accuracy" in training_history:
                metric_name = "Accuracy"
                metric_data = training_history["accuracy"]
            elif "r2_score" in training_history:
                metric_name = "R² Score"
                metric_data = training_history["r2_score"]
            else:
                metric_name = "Performance"
                metric_data = [1.0 - loss for loss in training_history["loss"]]  # Inverse loss
            
            axes[1].plot(epochs, metric_data, 
                        color=self.colors["classical"], linewidth=2, label=metric_name)
            axes[1].set_title(f"Model {metric_name}", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel(metric_name)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Plot quantum metrics if available
            if show_quantum_metrics and n_plots > 2 and "quantum_fidelity" in training_history:
                axes[2].plot(epochs, training_history["quantum_fidelity"], 
                            color=self.colors["quantum"], linewidth=2, label="Quantum Fidelity")
                axes[2].set_title("Quantum Fidelity", fontsize=14, fontweight='bold')
                axes[2].set_xlabel("Epoch")
                axes[2].set_ylabel("Fidelity")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training dynamics plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot training dynamics: {e}")
            raise VisualizationError(f"Training dynamics plotting failed: {e}")
    
    def plot_benchmark_comparison(
        self,
        benchmark_results: List[Dict[str, Any]],
        metric: str = "accuracy",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot benchmark comparison between QMANN and baseline models.
        
        Args:
            benchmark_results: List of benchmark result dictionaries
            metric: Metric to compare (accuracy, mse, training_time, etc.)
            save_path: Optional path to save the plot
        """
        try:
            # Organize data
            models = []
            datasets = []
            values = []
            model_types = []
            
            for result in benchmark_results:
                if metric in result and result[metric] is not None:
                    models.append(result["model_name"])
                    datasets.append(result["dataset_name"])
                    values.append(result[metric])
                    
                    if "QMANN" in result["model_name"]:
                        model_types.append("QMANN")
                    else:
                        model_types.append("Baseline")
            
            if not values:
                self.logger.warning(f"No data available for metric: {metric}")
                return
            
            # Create DataFrame for plotting
            import pandas as pd
            df = pd.DataFrame({
                "Model": models,
                "Dataset": datasets,
                "Value": values,
                "Type": model_types
            })
            
            # Create grouped bar plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot grouped bars
            datasets_unique = df["Dataset"].unique()
            x = np.arange(len(datasets_unique))
            width = 0.35
            
            qmann_values = []
            baseline_values = []
            
            for dataset in datasets_unique:
                dataset_data = df[df["Dataset"] == dataset]
                qmann_data = dataset_data[dataset_data["Type"] == "QMANN"]
                baseline_data = dataset_data[dataset_data["Type"] == "Baseline"]
                
                qmann_val = qmann_data["Value"].mean() if not qmann_data.empty else 0
                baseline_val = baseline_data["Value"].mean() if not baseline_data.empty else 0
                
                qmann_values.append(qmann_val)
                baseline_values.append(baseline_val)
            
            bars1 = ax.bar(x - width/2, qmann_values, width, 
                          label='QMANN', color=self.colors["qmann"], alpha=0.8)
            bars2 = ax.bar(x + width/2, baseline_values, width,
                          label='Baseline', color=self.colors["baseline"], alpha=0.8)
            
            # Customize plot
            ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_unique, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=10)
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Benchmark comparison plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot benchmark comparison: {e}")
            raise VisualizationError(f"Benchmark comparison plotting failed: {e}")
    
    def plot_quantum_memory_usage(
        self,
        memory_banks: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize quantum memory bank usage and fidelity.
        
        Args:
            memory_banks: List of memory bank information
            save_path: Optional path to save the plot
        """
        try:
            if not memory_banks:
                self.logger.warning("No memory bank data available")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Memory usage plot
            bank_ids = [f"Bank {i}" for i in range(len(memory_banks))]
            usage_values = [bank.get("usage", 0.0) for bank in memory_banks]
            fidelity_values = [bank.get("fidelity", 0.0) for bank in memory_banks]
            
            # Bar plot for memory usage
            bars = ax1.bar(bank_ids, usage_values, color=self.colors["quantum"], alpha=0.7)
            ax1.set_title("Quantum Memory Bank Usage", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Memory Bank")
            ax1.set_ylabel("Usage (%)")
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Add usage labels
            for bar, usage in zip(bars, usage_values):
                height = bar.get_height()
                ax1.annotate(f'{usage:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # Fidelity plot
            ax2.plot(bank_ids, fidelity_values, 'o-', color=self.colors["qmann"], 
                    linewidth=2, markersize=8, label="Fidelity")
            ax2.set_title("Quantum Memory Fidelity", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Memory Bank")
            ax2.set_ylabel("Fidelity")
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add fidelity labels
            for i, fidelity in enumerate(fidelity_values):
                ax2.annotate(f'{fidelity:.3f}',
                           xy=(i, fidelity),
                           xytext=(0, 10),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Quantum memory usage plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot quantum memory usage: {e}")
            raise VisualizationError(f"Quantum memory usage plotting failed: {e}")
    
    def plot_quantum_state(
        self,
        statevector: Union[np.ndarray, 'Statevector'],
        title: str = "Quantum State",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize quantum state using Q-sphere representation.
        
        Args:
            statevector: Quantum state vector
            title: Plot title
            save_path: Optional path to save the plot
        """
        if not QISKIT_AVAILABLE:
            self.logger.warning("Qiskit not available for quantum state visualization")
            return
        
        try:
            # Convert to Qiskit Statevector if needed
            if isinstance(statevector, np.ndarray):
                statevector = Statevector(statevector)
            
            # Create Q-sphere plot
            fig = plot_state_qsphere(statevector, title=title)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Quantum state plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot quantum state: {e}")
            raise VisualizationError(f"Quantum state plotting failed: {e}")
    
    def plot_circuit_depth_analysis(
        self,
        circuit_depths: List[int],
        gate_counts: List[int],
        fidelities: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Analyze relationship between circuit depth, gate count, and fidelity.
        
        Args:
            circuit_depths: List of circuit depths
            gate_counts: List of gate counts
            fidelities: List of corresponding fidelities
            save_path: Optional path to save the plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Depth vs Fidelity
            ax1.scatter(circuit_depths, fidelities, color=self.colors["quantum"], alpha=0.7)
            ax1.set_xlabel("Circuit Depth")
            ax1.set_ylabel("Fidelity")
            ax1.set_title("Circuit Depth vs Fidelity")
            ax1.grid(True, alpha=0.3)
            
            # Gate Count vs Fidelity
            ax2.scatter(gate_counts, fidelities, color=self.colors["hybrid"], alpha=0.7)
            ax2.set_xlabel("Gate Count")
            ax2.set_ylabel("Fidelity")
            ax2.set_title("Gate Count vs Fidelity")
            ax2.grid(True, alpha=0.3)
            
            # Depth vs Gate Count
            ax3.scatter(circuit_depths, gate_counts, color=self.colors["classical"], alpha=0.7)
            ax3.set_xlabel("Circuit Depth")
            ax3.set_ylabel("Gate Count")
            ax3.set_title("Circuit Depth vs Gate Count")
            ax3.grid(True, alpha=0.3)
            
            # Fidelity distribution
            ax4.hist(fidelities, bins=20, color=self.colors["qmann"], alpha=0.7, edgecolor='black')
            ax4.set_xlabel("Fidelity")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Fidelity Distribution")
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Circuit analysis plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot circuit analysis: {e}")
            raise VisualizationError(f"Circuit analysis plotting failed: {e}")
    
    def create_interactive_dashboard(
        self,
        training_history: Dict[str, List[float]],
        benchmark_results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            training_history: Training metrics over time
            benchmark_results: Benchmark comparison results
            save_path: Optional path to save the HTML dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Training Loss", "Performance Metrics", 
                              "Model Comparison", "Quantum Metrics"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs = list(range(1, len(training_history["loss"]) + 1))
            
            # Training loss
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history["loss"],
                          mode='lines+markers', name='Training Loss',
                          line=dict(color=self.colors["qmann"], width=3)),
                row=1, col=1
            )
            
            # Performance metrics
            if "accuracy" in training_history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history["accuracy"],
                              mode='lines+markers', name='Accuracy',
                              line=dict(color=self.colors["classical"], width=3)),
                    row=1, col=2
                )
            
            # Model comparison (if benchmark results available)
            if benchmark_results:
                models = [r["model_name"] for r in benchmark_results]
                accuracies = [r.get("accuracy", 0) for r in benchmark_results]
                
                fig.add_trace(
                    go.Bar(x=models, y=accuracies, name='Model Accuracy',
                          marker_color=[self.colors["qmann"] if "QMANN" in m 
                                      else self.colors["baseline"] for m in models]),
                    row=2, col=1
                )
            
            # Quantum metrics
            if "quantum_fidelity" in training_history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history["quantum_fidelity"],
                              mode='lines+markers', name='Quantum Fidelity',
                              line=dict(color=self.colors["quantum"], width=3)),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="QMANN Training and Performance Dashboard",
                showlegend=True,
                height=800,
                template="plotly_white"
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_xaxes(title_text="Model", row=2, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)
            fig.update_yaxes(title_text="Fidelity", row=2, col=2)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Interactive dashboard saved to {save_path}")
            
            fig.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive dashboard: {e}")
            raise VisualizationError(f"Interactive dashboard creation failed: {e}")
    
    def plot_error_mitigation_effectiveness(
        self,
        error_rates: List[float],
        mitigated_errors: List[float],
        methods: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot error mitigation effectiveness.
        
        Args:
            error_rates: Original error rates
            mitigated_errors: Error rates after mitigation
            methods: Error mitigation methods used
            save_path: Optional path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Before/After comparison
            x = np.arange(len(methods))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, error_rates, width, 
                           label='Original Error', color=self.colors["baseline"], alpha=0.7)
            bars2 = ax1.bar(x + width/2, mitigated_errors, width,
                           label='Mitigated Error', color=self.colors["qmann"], alpha=0.7)
            
            ax1.set_xlabel('Mitigation Method')
            ax1.set_ylabel('Error Rate')
            ax1.set_title('Error Mitigation Effectiveness')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Improvement percentage
            improvements = [(orig - mit) / orig * 100 for orig, mit in zip(error_rates, mitigated_errors)]
            bars3 = ax2.bar(methods, improvements, color=self.colors["quantum"], alpha=0.7)
            ax2.set_xlabel('Mitigation Method')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Error Reduction Percentage')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add improvement labels
            for bar, improvement in zip(bars3, improvements):
                height = bar.get_height()
                ax2.annotate(f'{improvement:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Error mitigation plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot error mitigation effectiveness: {e}")
            raise VisualizationError(f"Error mitigation plotting failed: {e}")
    
    def save_all_plots(self, output_dir: str, prefix: str = "qmann") -> None:
        """
        Save all generated plots to specified directory.
        
        Args:
            output_dir: Output directory for plots
            prefix: Filename prefix for saved plots
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"All plots will be saved to {output_dir} with prefix '{prefix}'")
    
    def set_style(self, style: str) -> None:
        """Set matplotlib style."""
        try:
            plt.style.use(style)
            self.logger.info(f"Plot style set to: {style}")
        except Exception as e:
            self.logger.warning(f"Failed to set style {style}: {e}")
    
    def get_color_scheme(self) -> Dict[str, str]:
        """Get current color scheme."""
        return self.colors.copy()
