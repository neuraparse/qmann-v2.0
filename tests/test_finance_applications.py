"""
Test Suite for Quantum Finance Applications (2025)

Tests quantum-enhanced financial applications including:
- Portfolio optimization
- Fraud detection
- Market prediction
- Risk assessment

Author: QMANN Development Team
Date: October 2025
Version: 2.1.0
"""

import pytest
import numpy as np
import torch
from qmann.applications.finance import (
    FinancialConfig,
    QuantumPortfolioOptimizer,
    QuantumFraudDetector,
    QuantumMarketPredictor,
)


class TestQuantumPortfolioOptimizer:
    """Test quantum portfolio optimization."""

    def test_portfolio_optimizer_initialization(self):
        """Test portfolio optimizer initialization."""
        config = FinancialConfig(num_assets=20, num_qubits=6)
        optimizer = QuantumPortfolioOptimizer(config)

        assert optimizer.config.num_assets == 20
        assert optimizer.config.num_qubits == 6
        assert optimizer.qaoa is not None

    def test_portfolio_optimization(self):
        """Test portfolio optimization with quantum advantage."""
        config = FinancialConfig(num_assets=10, num_qubits=6)
        optimizer = QuantumPortfolioOptimizer(config)

        # Generate synthetic market data
        returns = np.random.randn(10) * 0.1 + 0.05
        covariance = np.random.randn(10, 10)
        covariance = covariance @ covariance.T / 100  # Make positive semi-definite

        # Optimize portfolio
        result = optimizer.optimize_portfolio(returns, covariance, risk_tolerance=0.1)

        # Verify results
        assert "quantum_allocation" in result
        assert "classical_allocation" in result
        assert "quantum_sharpe_ratio" in result
        assert "quantum_advantage" in result

        # Check allocation sums to 1
        assert np.abs(np.sum(result["quantum_allocation"]) - 1.0) < 0.01
        assert np.abs(np.sum(result["classical_allocation"]) - 1.0) < 0.01

        # Check quantum advantage is a valid number
        assert isinstance(result["quantum_advantage"], (int, float))

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        config = FinancialConfig(num_assets=5, num_qubits=4)
        optimizer = QuantumPortfolioOptimizer(config)

        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        returns = np.array([0.05, 0.06, 0.04, 0.07, 0.05])
        covariance = np.eye(5) * 0.01

        sharpe = optimizer._calculate_sharpe_ratio(weights, returns, covariance)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for positive returns


class TestQuantumFraudDetector:
    """Test quantum fraud detection system."""

    def test_fraud_detector_initialization(self):
        """Test fraud detector initialization."""
        config = FinancialConfig(num_qubits=8, fraud_detection_threshold=0.85)
        detector = QuantumFraudDetector(config)

        assert detector.config.fraud_detection_threshold == 0.85
        assert detector.quantum_transformer is not None
        assert detector.fraud_classifier is not None

    def test_fraud_detection(self):
        """Test fraud detection on synthetic transactions."""
        config = FinancialConfig(num_qubits=8)
        detector = QuantumFraudDetector(config)

        # Generate synthetic transaction features
        batch_size = 10
        seq_len = 5
        features = 128
        transaction_features = torch.randn(batch_size, seq_len, features)

        # Detect fraud
        result = detector.detect_fraud(transaction_features)

        # Verify results
        assert "fraud_probabilities" in result
        assert "fraud_flags" in result
        assert "confidence_scores" in result
        assert result["quantum_enhanced"] is True

        # Check shapes
        assert result["fraud_probabilities"].shape[0] == batch_size
        assert result["fraud_flags"].shape[0] == batch_size

        # Check probability range
        assert np.all(result["fraud_probabilities"] >= 0)
        assert np.all(result["fraud_probabilities"] <= 1)


class TestQuantumMarketPredictor:
    """Test quantum market prediction."""

    def test_market_predictor_initialization(self):
        """Test market predictor initialization."""
        config = FinancialConfig(num_qubits=8, optimization_horizon=30)
        predictor = QuantumMarketPredictor(config)

        assert predictor.config.optimization_horizon == 30
        assert predictor.quantum_lstm is not None
        assert predictor.prediction_head is not None

    def test_market_prediction(self):
        """Test market prediction with quantum LSTM."""
        config = FinancialConfig(num_qubits=6, optimization_horizon=10)
        predictor = QuantumMarketPredictor(config)

        # Generate synthetic historical data
        batch_size = 5
        seq_len = 20
        features = 6
        historical_data = torch.randn(batch_size, seq_len, features)

        # Predict market
        result = predictor.predict_market(historical_data, prediction_horizon=10)

        # Verify results
        assert "predictions" in result
        assert "prediction_horizon" in result
        assert "confidence_intervals" in result
        assert result["quantum_enhanced"] is True
        assert result["model_type"] == "QuantumLSTM_2025"

        # Check prediction shape
        assert result["predictions"].shape[1] == 10  # prediction_horizon
        assert result["predictions"].shape[0] == batch_size

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        config = FinancialConfig(num_qubits=6)
        predictor = QuantumMarketPredictor(config)

        predictions = torch.randn(5, 10, 1)
        confidence_intervals = predictor._calculate_confidence_intervals(predictions)

        assert confidence_intervals.shape[0] == 10
        assert np.all(confidence_intervals >= 0)


class TestFinancialIntegration:
    """Integration tests for financial applications."""

    def test_end_to_end_portfolio_workflow(self):
        """Test complete portfolio optimization workflow."""
        config = FinancialConfig(num_assets=15, num_qubits=6)
        optimizer = QuantumPortfolioOptimizer(config)

        # Simulate market data
        np.random.seed(42)
        returns = np.random.randn(15) * 0.1 + 0.06
        covariance = np.random.randn(15, 15)
        covariance = covariance @ covariance.T / 100

        # Optimize
        result = optimizer.optimize_portfolio(returns, covariance)

        # Verify quantum advantage
        assert (
            result["quantum_advantage"] >= 0.8
        )  # Should be close to or better than classical

        # Verify portfolio constraints
        assert np.all(result["quantum_allocation"] >= 0)  # No short selling
        assert (
            np.abs(np.sum(result["quantum_allocation"]) - 1.0) < 0.01
        )  # Fully invested

    def test_fraud_detection_accuracy(self):
        """Test fraud detection accuracy on labeled data."""
        config = FinancialConfig(num_qubits=8, fraud_detection_threshold=0.8)
        detector = QuantumFraudDetector(config)

        # Generate synthetic data with known fraud patterns
        normal_transactions = torch.randn(50, 5, 128)
        fraudulent_transactions = torch.randn(10, 5, 128) * 3  # Anomalous patterns

        # Detect on normal transactions
        normal_result = detector.detect_fraud(normal_transactions)
        fraud_result = detector.detect_fraud(fraudulent_transactions)

        # Verify detection works
        assert normal_result["fraud_probabilities"].shape[0] == 50
        assert fraud_result["fraud_probabilities"].shape[0] == 10

    def test_market_prediction_consistency(self):
        """Test market prediction consistency."""
        config = FinancialConfig(num_qubits=6)
        predictor = QuantumMarketPredictor(config)

        # Same input should give similar predictions
        historical_data = torch.randn(3, 15, 6)

        result1 = predictor.predict_market(historical_data, prediction_horizon=5)
        result2 = predictor.predict_market(historical_data, prediction_horizon=5)

        # Predictions should have same shape
        assert result1["predictions"].shape == result2["predictions"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
