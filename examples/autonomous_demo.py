#!/usr/bin/env python3
"""
Autonomous Systems Coordination Demo

Demonstrates quantum-enhanced coordination for autonomous vehicles, drones,
and robotic systems using QMANN's quantum memory capabilities.
"""

import sys
import os
import time
import random
import numpy as np
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qmann.core.config import QMANNConfig
from qmann.applications.autonomous import AutonomousCoordination, AgentState


def generate_synthetic_agent_state(agent_id: str, agent_type: str = "drone") -> AgentState:
    """Generate synthetic agent state data."""
    
    # Different agent types have different characteristics
    agent_configs = {
        "drone": {
            "max_speed": 15.0,  # m/s
            "max_altitude": 100.0,  # m
            "communication_range": 200.0,  # m
            "battery_capacity": 1.0
        },
        "vehicle": {
            "max_speed": 25.0,  # m/s
            "max_altitude": 0.0,  # ground level
            "communication_range": 300.0,  # m
            "battery_capacity": 1.0
        },
        "robot": {
            "max_speed": 5.0,  # m/s
            "max_altitude": 0.0,  # ground level
            "communication_range": 100.0,  # m
            "battery_capacity": 1.0
        }
    }
    
    config = agent_configs.get(agent_type, agent_configs["drone"])
    
    # Random position within operational area
    position = (
        random.uniform(-100, 100),  # x
        random.uniform(-100, 100),  # y
        random.uniform(0, config["max_altitude"]) if config["max_altitude"] > 0 else 0.0  # z
    )
    
    # Random velocity within limits
    velocity = (
        random.uniform(-config["max_speed"], config["max_speed"]),
        random.uniform(-config["max_speed"], config["max_speed"]),
        random.uniform(-5, 5) if config["max_altitude"] > 0 else 0.0
    )
    
    # Random orientation
    orientation = (
        random.uniform(0, 2 * np.pi),  # roll
        random.uniform(0, 2 * np.pi),  # pitch
        random.uniform(0, 2 * np.pi)   # yaw
    )
    
    # Sensor data
    sensor_data = {
        "lidar_range": random.uniform(50, 200),
        "camera_visibility": random.uniform(0.7, 1.0),
        "gps_accuracy": random.uniform(0.5, 2.0),
        "imu_stability": random.uniform(0.8, 1.0),
        "obstacle_distance": random.uniform(5, 50)
    }
    
    # Mission status
    mission_statuses = ["idle", "moving", "coordinating", "mission_active", "returning"]
    mission_status = random.choice(mission_statuses)
    
    # Battery level
    battery_level = random.uniform(0.3, 1.0)
    
    return AgentState(
        agent_id=agent_id,
        position=position,
        velocity=velocity,
        orientation=orientation,
        sensor_data=sensor_data,
        mission_status=mission_status,
        battery_level=battery_level,
        communication_range=config["communication_range"],
        last_update=time.time()
    )


def demonstrate_basic_coordination():
    """Demonstrate basic multi-agent coordination."""
    print("=" * 60)
    print("BASIC MULTI-AGENT COORDINATION DEMO")
    print("=" * 60)
    
    # Create configuration
    config = QMANNConfig()
    
    # Initialize coordination system
    print("Initializing autonomous coordination system...")
    coordination_system = AutonomousCoordination(
        config=config,
        state_features=20,
        max_agents=10,
        coordination_range=150.0
    )
    coordination_system.initialize()
    print("✓ Coordination system initialized")
    
    # Generate agent states
    agent_types = ["drone", "vehicle", "robot"]
    agent_states = []
    
    print("\nGenerating synthetic agent states...")
    for i in range(5):
        agent_type = random.choice(agent_types)
        agent_id = f"{agent_type.upper()}_{i+1:03d}"
        agent_state = generate_synthetic_agent_state(agent_id, agent_type)
        agent_states.append(agent_state)
        print(f"  ✓ Generated state for {agent_id} at position "
              f"({agent_state.position[0]:.1f}, {agent_state.position[1]:.1f}, {agent_state.position[2]:.1f})")
    
    # Define mission objectives
    mission_objectives = {
        "mission_type": "area_coverage",
        "target_area": {"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50},
        "coordination_priority": "efficiency",
        "safety_margin": 10.0
    }
    
    # Coordinate agents
    print(f"\nCoordinating {len(agent_states)} agents...")
    coordination_decisions = coordination_system.coordinate_agents(
        agent_states=agent_states,
        mission_objectives=mission_objectives,
        use_quantum_memory=True
    )
    
    # Display coordination results
    print(f"\n🤖 COORDINATION DECISIONS")
    for decision in coordination_decisions:
        print(f"\nAgent: {decision.agent_id}")
        print(f"  Action: {decision.action_type}")
        print(f"  Target Position: ({decision.target_position[0]:.1f}, "
              f"{decision.target_position[1]:.1f}, {decision.target_position[2]:.1f})")
        print(f"  Priority Level: {decision.priority_level}/10")
        print(f"  Confidence: {decision.confidence_score:.3f}")
        print(f"  Quantum Coherence: {decision.quantum_coherence:.3f}")
        print(f"  Coordination Partners: {len(decision.coordination_partners)}")
        if decision.coordination_partners:
            print(f"    Partners: {', '.join(decision.coordination_partners[:3])}")
    
    return coordination_system, coordination_decisions


def demonstrate_swarm_behavior_prediction():
    """Demonstrate swarm behavior prediction."""
    print("\n" + "=" * 60)
    print("SWARM BEHAVIOR PREDICTION DEMO")
    print("=" * 60)
    
    # Create configuration
    config = QMANNConfig()
    
    # Initialize coordination system
    coordination_system = AutonomousCoordination(
        config=config,
        state_features=20,
        max_agents=15,
        coordination_range=100.0
    )
    coordination_system.initialize()
    
    # Generate larger swarm
    agent_states = []
    print("Generating swarm of autonomous agents...")
    
    for i in range(8):
        agent_type = "drone" if i < 5 else "vehicle"
        agent_id = f"SWARM_{i+1:03d}"
        agent_state = generate_synthetic_agent_state(agent_id, agent_type)
        agent_states.append(agent_state)
    
    print(f"  ✓ Generated {len(agent_states)} agents for swarm simulation")
    
    # Define scenario parameters
    scenario_parameters = {
        "scenario_type": "search_and_rescue",
        "search_area": {"x_min": -200, "x_max": 200, "y_min": -200, "y_max": 200},
        "target_locations": [(50, 75), (-30, -45), (120, -80)],
        "weather_conditions": "moderate_wind",
        "time_limit": 3600  # seconds
    }
    
    # Predict swarm behavior
    print(f"\nPredicting swarm behavior over 10 time steps...")
    behavior_prediction = coordination_system.predict_swarm_behavior(
        current_states=agent_states,
        prediction_horizon=10,
        scenario_parameters=scenario_parameters
    )
    
    # Display prediction results
    print(f"\n🔮 SWARM BEHAVIOR PREDICTION")
    print(f"Prediction Horizon: {behavior_prediction['prediction_horizon']} time steps")
    print(f"Emergent Patterns Detected: {len(behavior_prediction['emergent_patterns'])}")
    
    # Show emergent patterns
    if behavior_prediction['emergent_patterns']:
        print(f"\n🌟 EMERGENT PATTERNS:")
        for pattern in behavior_prediction['emergent_patterns']:
            print(f"  - {pattern['type'].title()} pattern at time step {pattern['time_step']}")
            if 'formation_type' in pattern:
                print(f"    Formation: {pattern['formation_type']}")
            if 'strength' in pattern:
                print(f"    Strength: {pattern['strength']:.3f}")
    
    # Show swarm cohesion evolution
    cohesion_evolution = behavior_prediction['swarm_cohesion_evolution']
    print(f"\n📊 SWARM COHESION EVOLUTION:")
    print(f"  Initial Cohesion: {cohesion_evolution[0]:.3f}")
    print(f"  Final Cohesion: {cohesion_evolution[-1]:.3f}")
    print(f"  Average Cohesion: {np.mean(cohesion_evolution):.3f}")
    
    # Show final swarm state
    final_state = behavior_prediction['final_swarm_state']
    print(f"\n🎯 PREDICTED FINAL STATE:")
    print(f"  Agent Count: {final_state['agent_count']}")
    print(f"  Average Position: ({final_state['average_position'][0]:.1f}, "
          f"{final_state['average_position'][1]:.1f}, {final_state['average_position'][2]:.1f})")
    print(f"  Swarm Spread: {final_state['swarm_spread']:.1f} meters")
    print(f"  Mission Progress: {final_state['mission_progress']:.1%}")
    
    return coordination_system, behavior_prediction


def demonstrate_formation_optimization():
    """Demonstrate swarm formation optimization."""
    print("\n" + "=" * 60)
    print("SWARM FORMATION OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Create configuration
    config = QMANNConfig()
    
    # Initialize coordination system
    coordination_system = AutonomousCoordination(
        config=config,
        state_features=20,
        max_agents=12,
        coordination_range=120.0
    )
    coordination_system.initialize()
    
    # Generate agents in suboptimal formation
    agent_states = []
    print("Generating agents in suboptimal formation...")
    
    for i in range(6):
        agent_id = f"FORM_{i+1:03d}"
        # Create clustered formation (suboptimal for coverage)
        angle = 2 * np.pi * i / 6
        radius = 20.0
        position = (
            radius * np.cos(angle),
            radius * np.sin(angle),
            random.uniform(10, 30)
        )
        
        agent_state = generate_synthetic_agent_state(agent_id, "drone")
        agent_state.position = position
        agent_states.append(agent_state)
        
        print(f"  ✓ Agent {agent_id} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
    
    # Test different formation optimizations
    formation_types = ["adaptive", "grid", "circular"]
    objective_functions = ["coverage", "efficiency", "safety"]
    
    print(f"\nTesting formation optimizations...")
    
    for formation_type in formation_types:
        for objective in objective_functions:
            print(f"\n📐 OPTIMIZING {formation_type.upper()} FORMATION FOR {objective.upper()}")
            
            optimization_result = coordination_system.optimize_swarm_formation(
                agent_states=agent_states,
                formation_type=formation_type,
                objective_function=objective
            )
            
            print(f"  Formation Type: {optimization_result['formation_type']}")
            print(f"  Objective: {optimization_result['objective_function']}")
            print(f"  Improvement Score: {optimization_result['improvement_score']:.3f}")
            
            # Show some optimal positions
            optimal_positions = optimization_result['optimal_positions']
            print(f"  Sample Optimal Positions:")
            for i, (agent_id, position) in enumerate(list(optimal_positions.items())[:3]):
                print(f"    {agent_id}: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
            
            # Show movement commands
            movement_commands = optimization_result['movement_commands']
            print(f"  Movement Commands Generated: {len(movement_commands)}")
    
    return coordination_system, optimization_result


def demonstrate_real_time_coordination():
    """Demonstrate real-time coordination simulation."""
    print("\n" + "=" * 60)
    print("REAL-TIME COORDINATION SIMULATION")
    print("=" * 60)
    
    # Create configuration
    config = QMANNConfig()
    
    # Initialize coordination system
    coordination_system = AutonomousCoordination(
        config=config,
        state_features=20,
        max_agents=8,
        coordination_range=100.0
    )
    coordination_system.initialize()
    
    # Generate initial agent states
    agent_states = []
    for i in range(4):
        agent_id = f"RT_{i+1:03d}"
        agent_state = generate_synthetic_agent_state(agent_id, "drone")
        agent_states.append(agent_state)
    
    print(f"Starting real-time simulation with {len(agent_states)} agents...")
    
    # Simulate multiple coordination cycles
    simulation_steps = 5
    mission_objectives = {
        "mission_type": "patrol",
        "patrol_area": {"x_min": -80, "x_max": 80, "y_min": -80, "y_max": 80},
        "coordination_frequency": 1.0  # seconds
    }
    
    for step in range(simulation_steps):
        print(f"\n⏱️  SIMULATION STEP {step + 1}/{simulation_steps}")
        
        # Coordinate agents
        decisions = coordination_system.coordinate_agents(
            agent_states=agent_states,
            mission_objectives=mission_objectives,
            use_quantum_memory=True
        )
        
        # Show coordination summary
        action_counts = {}
        for decision in decisions:
            action_counts[decision.action_type] = action_counts.get(decision.action_type, 0) + 1
        
        print(f"  Actions: {dict(action_counts)}")
        
        # Simulate agent movement based on decisions
        for i, (agent_state, decision) in enumerate(zip(agent_states, decisions)):
            # Simple movement simulation
            new_position = tuple(
                np.array(agent_state.position) + 
                np.array(decision.target_velocity) * 0.5  # 0.5 second time step
            )
            agent_states[i].position = new_position
            agent_states[i].velocity = decision.target_velocity
            agent_states[i].last_update = time.time()
        
        # Show agent positions
        print(f"  Agent Positions:")
        for agent_state in agent_states:
            print(f"    {agent_state.agent_id}: "
                  f"({agent_state.position[0]:.1f}, {agent_state.position[1]:.1f})")
        
        # Brief pause to simulate real-time
        time.sleep(0.1)
    
    # Get final statistics
    stats = coordination_system.get_coordination_statistics()
    
    print(f"\n📊 SIMULATION STATISTICS")
    print(f"Total Coordination Cycles: {stats['total_coordination_cycles']}")
    print(f"Average Agents per Cycle: {stats['average_agents_per_cycle']:.1f}")
    print(f"Action Distribution: {stats['action_distribution']}")
    print(f"Successful Coordinations: {stats['swarm_metrics']['successful_coordinations']}")
    print(f"Collision Avoidances: {stats['swarm_metrics']['collision_avoidances']}")
    
    return coordination_system, stats


def main():
    """Main demo function."""
    print("🤖 QMANN Autonomous Systems Coordination Demo")
    print("Quantum-Enhanced Multi-Agent Coordination and Swarm Intelligence")
    print()
    
    try:
        # Run all demonstrations
        coord_system, decisions = demonstrate_basic_coordination()
        coord_system, prediction = demonstrate_swarm_behavior_prediction()
        coord_system, optimization = demonstrate_formation_optimization()
        coord_system, stats = demonstrate_real_time_coordination()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Achievements:")
        print("✓ Multi-agent coordination with quantum memory")
        print("✓ Swarm behavior prediction and pattern detection")
        print("✓ Formation optimization for different objectives")
        print("✓ Real-time coordination simulation")
        print()
        print("The QMANN autonomous coordination system demonstrates:")
        print("• Quantum-enhanced decision making")
        print("• Emergent swarm intelligence")
        print("• Adaptive formation control")
        print("• Real-time collision avoidance")
        print()
        print("Applications include:")
        print("• Autonomous vehicle coordination")
        print("• Drone swarm operations")
        print("• Robotic team coordination")
        print("• Search and rescue missions")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
