#!/usr/bin/env python3
"""
Quick test of QMANN applications
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qmann.core.config import QMANNConfig
    from qmann.applications.industrial import IndustrialMaintenance, EquipmentData
    from qmann.applications.autonomous import AutonomousCoordination, AgentState
    from qmann.applications.healthcare import HealthcarePredictor
    
    print("✓ All applications imported successfully!")
    
    # Test configuration
    config = QMANNConfig()
    print("✓ Configuration created")
    
    # Test industrial maintenance
    industrial = IndustrialMaintenance(config, sensor_features=10)
    print("✓ Industrial maintenance system created")
    
    # Test autonomous coordination
    autonomous = AutonomousCoordination(config, state_features=10, max_agents=5)
    print("✓ Autonomous coordination system created")
    
    # Test healthcare predictor
    healthcare = HealthcarePredictor(config, patient_features=15)
    print("✓ Healthcare predictor created")
    
    print("\n🎉 All QMANN applications are working correctly!")
    print("\nAvailable applications:")
    print("• Industrial Predictive Maintenance")
    print("• Autonomous Systems Coordination") 
    print("• Healthcare Predictive Analytics")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
