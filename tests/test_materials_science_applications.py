"""
Test Suite for Quantum Materials Science Applications (2025)

Tests quantum-enhanced materials science applications including:
- Material property prediction
- Crystal structure optimization
- Battery material design
- Catalyst design

Author: QMANN Development Team
Date: October 2025
Version: 2.1.0
"""

import pytest
import numpy as np
import torch
from qmann.applications.materials_science import (
    MaterialsScienceConfig,
    QuantumMaterialPropertyPredictor,
    QuantumCrystalStructureOptimizer,
    QuantumBatteryMaterialDesigner,
)


class TestQuantumMaterialPropertyPredictor:
    """Test quantum material property prediction."""

    def test_property_predictor_initialization(self):
        """Test material property predictor initialization."""
        config = MaterialsScienceConfig(num_qubits=16, max_atoms=200)
        predictor = QuantumMaterialPropertyPredictor(config)

        assert predictor.config.num_qubits == 16
        assert predictor.material_encoder is not None
        assert len(predictor.property_predictors) == 15

    def test_property_prediction(self):
        """Test material property prediction."""
        config = MaterialsScienceConfig(num_qubits=12)
        predictor = QuantumMaterialPropertyPredictor(config)

        # Generate synthetic material features
        batch_size = 3
        seq_len = 25
        features = 512
        material_features = torch.randn(batch_size, seq_len, features)

        # Predict properties
        result = predictor.predict_properties(material_features)

        # Verify results
        assert "properties" in result
        assert "material_quality_score" in result
        assert "application_suitability" in result
        assert result["quantum_enhanced"] is True

        # Check all properties are predicted
        expected_properties = [
            "band_gap",
            "formation_energy",
            "elastic_modulus",
            "thermal_conductivity",
            "electrical_conductivity",
            "magnetic_moment",
            "density",
            "hardness",
            "melting_point",
            "stability",
            "catalytic_activity",
            "ionic_conductivity",
            "dielectric_constant",
            "refractive_index",
            "superconducting_tc",
        ]
        for prop in expected_properties:
            assert prop in result["properties"]

    def test_material_quality_calculation(self):
        """Test material quality score calculation."""
        config = MaterialsScienceConfig(num_qubits=12)
        predictor = QuantumMaterialPropertyPredictor(config)

        # Create mock properties
        properties = {
            "stability": np.array([0.8]),
            "formation_energy": np.array([-0.5]),
            "band_gap": np.array([1.5]),
        }

        quality_score = predictor._calculate_material_quality(properties)

        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1

    def test_application_determination(self):
        """Test application suitability determination."""
        config = MaterialsScienceConfig(num_qubits=12)
        predictor = QuantumMaterialPropertyPredictor(config)

        # Semiconductor properties
        semiconductor_props = {
            "band_gap": np.array([1.5]),
            "ionic_conductivity": np.array([0.3]),
            "catalytic_activity": np.array([0.4]),
            "superconducting_tc": np.array([50]),
            "elastic_modulus": np.array([150]),
            "hardness": np.array([7]),
        }

        applications = predictor._determine_applications(semiconductor_props)

        assert isinstance(applications, list)
        assert len(applications) > 0
        assert "semiconductor" in applications


class TestQuantumCrystalStructureOptimizer:
    """Test quantum crystal structure optimization."""

    def test_structure_optimizer_initialization(self):
        """Test structure optimizer initialization."""
        config = MaterialsScienceConfig(num_qubits=16, structure_optimization_steps=100)
        optimizer = QuantumCrystalStructureOptimizer(config)

        assert optimizer.config.structure_optimization_steps == 100
        assert optimizer.structure_optimizer is not None
        assert optimizer.qaoa_optimizer is not None

    def test_structure_optimization(self):
        """Test crystal structure optimization."""
        config = MaterialsScienceConfig(num_qubits=12, structure_optimization_steps=10)
        optimizer = QuantumCrystalStructureOptimizer(config)

        # Create initial structure (10 atoms)
        initial_structure = np.random.randn(10, 3)

        # Optimize structure
        result = optimizer.optimize_structure(initial_structure)

        # Verify results
        assert "optimized_structure" in result
        assert "initial_structure" in result
        assert "final_energy" in result
        assert "energy_improvement" in result
        assert result["quantum_advantage"] is True

        # Check structure shape - may be adjusted based on num_qubits
        assert result["optimized_structure"].shape[1] == 3  # 3D coordinates
        assert result["optimized_structure"].shape[0] > 0  # Has atoms

    def test_structure_to_quantum_params_conversion(self):
        """Test structure to quantum parameters conversion."""
        config = MaterialsScienceConfig(num_qubits=12)
        optimizer = QuantumCrystalStructureOptimizer(config)

        structure = np.random.randn(8, 3)
        params = optimizer._structure_to_quantum_params(structure)

        assert params.shape[0] == config.num_qubits * 3
        assert np.all(np.abs(params) <= 1.0)  # Normalized

    def test_quantum_params_to_structure_conversion(self):
        """Test quantum parameters to structure conversion."""
        config = MaterialsScienceConfig(num_qubits=12)
        optimizer = QuantumCrystalStructureOptimizer(config)

        params = np.random.randn(config.num_qubits * 3)
        structure = optimizer._quantum_params_to_structure(params)

        assert structure.shape[1] == 3  # 3D coordinates
        assert structure.shape[0] == config.num_qubits


class TestQuantumBatteryMaterialDesigner:
    """Test quantum battery material design."""

    def test_battery_designer_initialization(self):
        """Test battery material designer initialization."""
        config = MaterialsScienceConfig(num_qubits=16)
        designer = QuantumBatteryMaterialDesigner(config)

        assert designer.battery_analyzer is not None
        assert len(designer.performance_predictors) == 9

    def test_battery_material_design(self):
        """Test battery material design."""
        config = MaterialsScienceConfig(num_qubits=12)
        designer = QuantumBatteryMaterialDesigner(config)

        # Generate synthetic material features
        batch_size = 2
        seq_len = 20
        features = 384
        material_features = torch.randn(batch_size, seq_len, features)

        # Design battery material
        result = designer.design_battery_material(
            material_features, target_application="electric_vehicle"
        )

        # Verify results
        assert "performance_metrics" in result
        assert "target_application" in result
        assert "suitability_score" in result
        assert result["quantum_enhanced"] is True

        # Check all performance metrics
        expected_metrics = [
            "capacity",
            "voltage",
            "cycle_life",
            "rate_capability",
            "safety",
            "cost_effectiveness",
            "energy_density",
            "power_density",
            "thermal_stability",
        ]
        for metric in expected_metrics:
            assert metric in result["performance_metrics"]

    def test_application_suitability_calculation(self):
        """Test application suitability calculation."""
        config = MaterialsScienceConfig(num_qubits=12)
        designer = QuantumBatteryMaterialDesigner(config)

        # Create mock performance metrics
        performance = {
            "energy_density": np.array([250]),
            "cycle_life": np.array([1500]),
            "safety": np.array([0.9]),
            "rate_capability": np.array([0.8]),
            "cost_effectiveness": np.array([0.7]),
        }

        # Test EV application
        ev_suitability = designer._calculate_application_suitability(
            performance, "electric_vehicle"
        )
        assert isinstance(ev_suitability, float)
        assert 0 <= ev_suitability <= 1

        # Test grid storage application
        grid_suitability = designer._calculate_application_suitability(
            performance, "grid_storage"
        )
        assert isinstance(grid_suitability, float)
        assert 0 <= grid_suitability <= 1

    def test_production_cost_estimation(self):
        """Test production cost estimation."""
        config = MaterialsScienceConfig(num_qubits=12)
        designer = QuantumBatteryMaterialDesigner(config)

        performance = {"energy_density": np.array([300]), "capacity": np.array([200])}

        cost = designer._estimate_production_cost(performance)

        assert isinstance(cost, float)
        assert cost > 0


class TestMaterialsScienceIntegration:
    """Integration tests for materials science applications."""

    def test_end_to_end_material_discovery(self):
        """Test complete material discovery workflow."""
        config = MaterialsScienceConfig(num_qubits=12)

        # Initialize components
        property_predictor = QuantumMaterialPropertyPredictor(config)
        structure_optimizer = QuantumCrystalStructureOptimizer(config)

        # Generate and optimize material
        material_features = torch.randn(1, 20, 512)
        initial_structure = np.random.randn(8, 3)

        # Predict properties
        properties = property_predictor.predict_properties(material_features)

        # Optimize structure
        optimized = structure_optimizer.optimize_structure(
            initial_structure, energy_function=None
        )

        # Verify workflow
        assert properties["material_quality_score"] >= 0
        assert optimized["energy_improvement"] >= 0
        assert len(properties["application_suitability"]) > 0

    def test_battery_material_screening(self):
        """Test battery material screening workflow."""
        config = MaterialsScienceConfig(num_qubits=12)

        property_predictor = QuantumMaterialPropertyPredictor(config)
        battery_designer = QuantumBatteryMaterialDesigner(config)

        # Screen material for battery application
        material_features = torch.randn(1, 20, 512)

        # Predict general properties
        properties = property_predictor.predict_properties(material_features)

        # Design for battery application
        battery_design = battery_designer.design_battery_material(
            material_features, target_application="electric_vehicle"
        )

        # Verify screening
        assert "ionic_conductivity" in properties["properties"]
        assert battery_design["suitability_score"] >= 0
        assert "safety_rating" in battery_design

    def test_multi_application_material_design(self):
        """Test material design for multiple applications."""
        config = MaterialsScienceConfig(num_qubits=12)
        battery_designer = QuantumBatteryMaterialDesigner(config)

        material_features = torch.randn(1, 20, 384)

        # Test different applications
        applications = ["electric_vehicle", "grid_storage", "portable"]
        results = {}

        for app in applications:
            result = battery_designer.design_battery_material(
                material_features, target_application=app
            )
            results[app] = result["suitability_score"]

        # Verify all applications evaluated
        assert len(results) == 3
        for score in results.values():
            assert 0 <= score <= 1

    def test_structure_optimization_convergence(self):
        """Test structure optimization convergence."""
        config = MaterialsScienceConfig(
            num_qubits=12,
            structure_optimization_steps=50,
            energy_convergence_threshold=1e-6,
        )
        optimizer = QuantumCrystalStructureOptimizer(config)

        initial_structure = np.random.randn(10, 3)
        result = optimizer.optimize_structure(initial_structure)

        # Check convergence
        assert "converged" in result
        assert "optimization_steps" in result
        assert result["optimization_steps"] <= config.structure_optimization_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
