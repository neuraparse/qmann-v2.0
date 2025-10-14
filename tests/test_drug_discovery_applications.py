"""
Test Suite for Quantum Drug Discovery Applications (2025)

Tests quantum-enhanced pharmaceutical applications including:
- Molecular property prediction
- Drug-target binding affinity
- Molecular generation
- ADMET prediction

Author: QMANN Development Team
Date: October 2025
Version: 2.1.0
"""

import pytest
import numpy as np
import torch
from qmann.applications.drug_discovery import (
    DrugDiscoveryConfig,
    QuantumMolecularPropertyPredictor,
    QuantumDrugTargetBindingPredictor,
    QuantumMolecularGenerator,
    QuantumADMETPredictor
)


class TestQuantumMolecularPropertyPredictor:
    """Test quantum molecular property prediction."""
    
    def test_property_predictor_initialization(self):
        """Test molecular property predictor initialization."""
        config = DrugDiscoveryConfig(num_qubits=12, max_molecular_size=100)
        predictor = QuantumMolecularPropertyPredictor(config)
        
        assert predictor.config.num_qubits == 12
        assert predictor.molecular_encoder is not None
        assert len(predictor.property_predictors) == 10
    
    def test_property_prediction(self):
        """Test molecular property prediction."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumMolecularPropertyPredictor(config)
        
        # Generate synthetic molecular features
        batch_size = 5
        seq_len = 20
        features = 256
        molecular_features = torch.randn(batch_size, seq_len, features)
        
        # Predict properties
        result = predictor.predict_properties(molecular_features)
        
        # Verify results
        assert 'properties' in result
        assert 'drug_likeness_score' in result
        assert 'quantum_enhanced' in result
        assert result['quantum_enhanced'] is True
        
        # Check all properties are predicted
        expected_properties = ['logP', 'solubility', 'binding_affinity', 'toxicity',
                              'bioavailability', 'clearance', 'half_life', 'permeability',
                              'stability', 'selectivity']
        for prop in expected_properties:
            assert prop in result['properties']
    
    def test_drug_likeness_calculation(self):
        """Test drug-likeness score calculation."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumMolecularPropertyPredictor(config)
        
        # Create mock properties
        properties = {
            'logP': np.array([2.5]),
            'solubility': np.array([0.5]),
            'toxicity': np.array([0.3]),
            'bioavailability': np.array([0.7])
        }
        
        drug_likeness = predictor._calculate_drug_likeness(properties)
        
        assert isinstance(drug_likeness, float)
        assert 0 <= drug_likeness <= 1


class TestQuantumDrugTargetBindingPredictor:
    """Test quantum drug-target binding prediction."""
    
    def test_binding_predictor_initialization(self):
        """Test binding predictor initialization."""
        config = DrugDiscoveryConfig(num_qubits=12, binding_affinity_threshold=-8.0)
        predictor = QuantumDrugTargetBindingPredictor(config)
        
        assert predictor.config.binding_affinity_threshold == -8.0
        assert predictor.drug_encoder is not None
        assert predictor.target_encoder is not None
        assert predictor.interaction_predictor is not None
    
    def test_binding_affinity_prediction(self):
        """Test binding affinity prediction."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumDrugTargetBindingPredictor(config)
        
        # Generate synthetic drug and target features
        batch_size = 3
        seq_len = 15
        features = 256
        drug_features = torch.randn(batch_size, seq_len, features)
        target_features = torch.randn(batch_size, seq_len, features)
        
        # Predict binding affinity
        result = predictor.predict_binding_affinity(drug_features, target_features)
        
        # Verify results
        assert 'binding_affinity' in result
        assert 'strong_binding' in result
        assert 'interaction_strength' in result
        assert result['quantum_enhanced'] is True
        
        # Check shapes
        assert result['binding_affinity'].shape[0] == batch_size
    
    def test_interaction_strength_analysis(self):
        """Test interaction strength analysis."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumDrugTargetBindingPredictor(config)
        
        drug_embedding = torch.randn(5, 256)
        target_embedding = torch.randn(5, 256)
        
        interaction_strength = predictor._analyze_interaction_strength(
            drug_embedding, target_embedding
        )
        
        assert isinstance(interaction_strength, float)
        assert -1 <= interaction_strength <= 1  # Cosine similarity range


class TestQuantumMolecularGenerator:
    """Test quantum molecular generation."""
    
    def test_molecular_generator_initialization(self):
        """Test molecular generator initialization."""
        config = DrugDiscoveryConfig(num_qubits=12, max_molecular_size=100)
        generator = QuantumMolecularGenerator(config)
        
        assert generator.config.max_molecular_size == 100
        assert generator.molecular_search is not None
        assert generator.molecular_decoder is not None
    
    def test_molecule_generation(self):
        """Test novel molecule generation."""
        config = DrugDiscoveryConfig(num_qubits=10, max_molecular_size=50)
        generator = QuantumMolecularGenerator(config)
        
        # Define target properties
        target_properties = {
            'logP': 2.5,
            'solubility': 0.6,
            'binding_affinity': -9.0
        }
        
        # Generate molecules
        result = generator.generate_molecules(target_properties, num_candidates=5)
        
        # Verify results
        assert 'generated_candidates' in result
        assert 'num_candidates' in result
        assert result['num_candidates'] == 5
        assert len(result['generated_candidates']) == 5
        assert result['quantum_advantage'] is True
        
        # Check each candidate
        for candidate in result['generated_candidates']:
            assert 'coordinates' in candidate
            assert 'quantum_state' in candidate
            assert candidate['generation_method'] == 'GroverDynamics_2025'


class TestQuantumADMETPredictor:
    """Test quantum ADMET prediction."""
    
    def test_admet_predictor_initialization(self):
        """Test ADMET predictor initialization."""
        config = DrugDiscoveryConfig(num_qubits=12, toxicity_threshold=0.5)
        predictor = QuantumADMETPredictor(config)
        
        assert predictor.config.toxicity_threshold == 0.5
        assert predictor.admet_encoder is not None
        assert len(predictor.admet_predictors) == 8
    
    def test_admet_prediction(self):
        """Test ADMET property prediction."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumADMETPredictor(config)
        
        # Generate synthetic molecular features
        batch_size = 4
        seq_len = 18
        features = 256
        molecular_features = torch.randn(batch_size, seq_len, features)
        
        # Predict ADMET
        result = predictor.predict_admet(molecular_features)
        
        # Verify results
        assert 'admet_properties' in result
        assert 'safety_score' in result
        assert 'safe_for_clinical_trials' in result
        assert 'toxicity_flags' in result
        assert result['quantum_enhanced'] is True
        
        # Check all ADMET properties
        expected_properties = ['absorption', 'distribution', 'metabolism', 'excretion',
                              'toxicity', 'hepatotoxicity', 'cardiotoxicity', 'mutagenicity']
        for prop in expected_properties:
            assert prop in result['admet_properties']
    
    def test_safety_score_calculation(self):
        """Test safety score calculation."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumADMETPredictor(config)
        
        # Create mock ADMET predictions
        admet_predictions = {
            'toxicity': np.array([0.3]),
            'hepatotoxicity': np.array([0.2]),
            'cardiotoxicity': np.array([0.25]),
            'mutagenicity': np.array([0.15])
        }
        
        safety_score = predictor._calculate_safety_score(admet_predictions)
        
        assert isinstance(safety_score, float)
        assert 0 <= safety_score <= 1
        assert safety_score > 0.5  # Should be safe with low toxicity values
    
    def test_toxicity_flag_identification(self):
        """Test toxicity flag identification."""
        config = DrugDiscoveryConfig(num_qubits=10)
        predictor = QuantumADMETPredictor(config)
        
        # High toxicity predictions
        high_toxicity = {
            'hepatotoxicity': np.array([0.8]),
            'cardiotoxicity': np.array([0.7]),
            'mutagenicity': np.array([0.6]),
            'toxicity': np.array([0.75])
        }
        
        flags = predictor._identify_toxicity_flags(high_toxicity)
        
        assert len(flags) > 0
        assert any('HEPATOTOXICITY' in flag for flag in flags)
        assert any('CARDIOTOXICITY' in flag for flag in flags)


class TestDrugDiscoveryIntegration:
    """Integration tests for drug discovery applications."""
    
    def test_end_to_end_drug_screening(self):
        """Test complete drug screening workflow."""
        config = DrugDiscoveryConfig(num_qubits=10)
        
        # Initialize all components
        property_predictor = QuantumMolecularPropertyPredictor(config)
        binding_predictor = QuantumDrugTargetBindingPredictor(config)
        admet_predictor = QuantumADMETPredictor(config)
        
        # Generate synthetic molecule
        molecular_features = torch.randn(1, 20, 256)
        target_features = torch.randn(1, 20, 256)
        
        # Screen molecule
        properties = property_predictor.predict_properties(molecular_features)
        binding = binding_predictor.predict_binding_affinity(molecular_features, target_features)
        admet = admet_predictor.predict_admet(molecular_features)
        
        # Verify all predictions
        assert properties['drug_likeness_score'] >= 0
        assert 'binding_affinity' in binding
        assert admet['safety_score'] >= 0
    
    def test_molecule_generation_and_evaluation(self):
        """Test molecule generation and evaluation pipeline."""
        config = DrugDiscoveryConfig(num_qubits=10, max_molecular_size=50)
        
        generator = QuantumMolecularGenerator(config)
        property_predictor = QuantumMolecularPropertyPredictor(config)
        
        # Generate molecules
        target_properties = {'logP': 2.5, 'solubility': 0.6}
        generated = generator.generate_molecules(target_properties, num_candidates=3)
        
        assert len(generated['generated_candidates']) == 3
        assert generated['quantum_advantage'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

