"""
QMANN Applications (2025 Global Industry Standards)

Real-world applications demonstrating the capabilities of
quantum memory-augmented neural networks across multiple industries.

Industry Applications:
- Finance: Portfolio optimization, fraud detection, market prediction
- Drug Discovery: Molecular property prediction, drug-target binding, ADMET
- Materials Science: Material discovery, crystal optimization, battery design
- Healthcare: Medical diagnosis, treatment planning, patient monitoring
- Industrial: Predictive maintenance, quality control, process optimization
- Autonomous Systems: Multi-agent coordination, path planning, decision making

Author: QMANN Development Team
Date: October 2025
Version: 2.1.0
"""

# Healthcare Applications
from .healthcare import HealthcarePredictor

# Industrial Applications
from .industrial import IndustrialMaintenance

# Autonomous Systems
from .autonomous import AutonomousCoordination

# Finance Applications (2025)
from .finance import (
    FinancialConfig,
    QuantumPortfolioOptimizer,
    QuantumFraudDetector,
    QuantumMarketPredictor
)

# Drug Discovery Applications (2025)
from .drug_discovery import (
    DrugDiscoveryConfig,
    QuantumMolecularPropertyPredictor,
    QuantumDrugTargetBindingPredictor,
    QuantumMolecularGenerator,
    QuantumADMETPredictor
)

# Materials Science Applications (2025)
from .materials_science import (
    MaterialsScienceConfig,
    QuantumMaterialPropertyPredictor,
    QuantumCrystalStructureOptimizer,
    QuantumBatteryMaterialDesigner
)

__all__ = [
    # Healthcare
    "HealthcarePredictor",

    # Industrial
    "IndustrialMaintenance",

    # Autonomous
    "AutonomousCoordination",

    # Finance (2025)
    "FinancialConfig",
    "QuantumPortfolioOptimizer",
    "QuantumFraudDetector",
    "QuantumMarketPredictor",

    # Drug Discovery (2025)
    "DrugDiscoveryConfig",
    "QuantumMolecularPropertyPredictor",
    "QuantumDrugTargetBindingPredictor",
    "QuantumMolecularGenerator",
    "QuantumADMETPredictor",

    # Materials Science (2025)
    "MaterialsScienceConfig",
    "QuantumMaterialPropertyPredictor",
    "QuantumCrystalStructureOptimizer",
    "QuantumBatteryMaterialDesigner",
]
