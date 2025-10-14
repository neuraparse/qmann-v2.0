-- QMANN Database Initialization Script
-- PostgreSQL schema for quantum memory-augmented neural networks

-- Create database if not exists
-- Note: This script assumes the database 'qmann' already exists

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS qmann;
CREATE SCHEMA IF NOT EXISTS quantum;
CREATE SCHEMA IF NOT EXISTS classical;
CREATE SCHEMA IF NOT EXISTS applications;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO qmann, quantum, classical, applications, monitoring, public;

-- ============================================================================
-- CORE QMANN TABLES
-- ============================================================================

-- Models table
CREATE TABLE IF NOT EXISTS qmann.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('quantum_lstm', 'hybrid_lstm', 'classical_lstm')),
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    config JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deprecated'))
);

-- Training sessions
CREATE TABLE IF NOT EXISTS qmann.training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'paused')),
    hyperparameters JSONB NOT NULL,
    metrics JSONB,
    quantum_backend VARCHAR(100),
    classical_backend VARCHAR(100),
    total_epochs INTEGER,
    current_epoch INTEGER DEFAULT 0,
    best_loss FLOAT,
    best_accuracy FLOAT,
    created_by VARCHAR(255)
);

-- Training metrics (time series data)
CREATE TABLE IF NOT EXISTS qmann.training_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES qmann.training_sessions(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    step INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(20) DEFAULT 'scalar' CHECK (metric_type IN ('scalar', 'histogram', 'image')),
    metadata JSONB
);

-- ============================================================================
-- QUANTUM COMPUTING TABLES
-- ============================================================================

-- Quantum backends
CREATE TABLE IF NOT EXISTS quantum.backends (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL,
    backend_type VARCHAR(20) NOT NULL CHECK (backend_type IN ('simulator', 'hardware')),
    num_qubits INTEGER NOT NULL,
    quantum_volume INTEGER,
    gate_error_rate FLOAT,
    readout_error_rate FLOAT,
    coherence_time_t1 FLOAT,
    coherence_time_t2 FLOAT,
    status VARCHAR(20) DEFAULT 'active',
    last_calibration TIMESTAMP WITH TIME ZONE,
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quantum circuits
CREATE TABLE IF NOT EXISTS quantum.circuits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id) ON DELETE CASCADE,
    circuit_name VARCHAR(255) NOT NULL,
    circuit_type VARCHAR(50) NOT NULL,
    num_qubits INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    gate_count INTEGER NOT NULL,
    qasm_code TEXT,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quantum jobs
CREATE TABLE IF NOT EXISTS quantum.jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    circuit_id UUID NOT NULL REFERENCES quantum.circuits(id) ON DELETE CASCADE,
    backend_id UUID NOT NULL REFERENCES quantum.backends(id),
    job_id VARCHAR(255) UNIQUE,
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    shots INTEGER NOT NULL DEFAULT 1024,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    queue_position INTEGER,
    estimated_start_time TIMESTAMP WITH TIME ZONE,
    results JSONB,
    error_message TEXT
);

-- Quantum memory states
CREATE TABLE IF NOT EXISTS quantum.memory_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id) ON DELETE CASCADE,
    memory_bank INTEGER NOT NULL,
    state_vector BYTEA,
    fidelity FLOAT,
    entanglement_measure FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Error mitigation results
CREATE TABLE IF NOT EXISTS quantum.error_mitigation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES quantum.jobs(id) ON DELETE CASCADE,
    method VARCHAR(50) NOT NULL,
    raw_results JSONB NOT NULL,
    mitigated_results JSONB NOT NULL,
    improvement_factor FLOAT,
    execution_time FLOAT,
    parameters JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- CLASSICAL ML TABLES
-- ============================================================================

-- Datasets
CREATE TABLE IF NOT EXISTS classical.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    dataset_type VARCHAR(50) NOT NULL,
    size_bytes BIGINT,
    num_samples INTEGER,
    num_features INTEGER,
    target_type VARCHAR(20) CHECK (target_type IN ('classification', 'regression', 'sequence')),
    file_path TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model checkpoints
CREATE TABLE IF NOT EXISTS classical.checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES qmann.training_sessions(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    model_state_dict BYTEA,
    optimizer_state_dict BYTEA,
    loss FLOAT,
    accuracy FLOAT,
    quantum_fidelity FLOAT,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- APPLICATION TABLES
-- ============================================================================

-- Healthcare predictions
CREATE TABLE IF NOT EXISTS applications.healthcare_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id),
    patient_id VARCHAR(255) NOT NULL,
    prediction_type VARCHAR(100) NOT NULL,
    input_features JSONB NOT NULL,
    prediction_result JSONB NOT NULL,
    confidence_score FLOAT,
    quantum_contribution FLOAT,
    risk_factors JSONB,
    recommendations JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Industrial maintenance predictions
CREATE TABLE IF NOT EXISTS applications.industrial_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id),
    equipment_id VARCHAR(255) NOT NULL,
    sensor_data JSONB NOT NULL,
    failure_probability FLOAT,
    time_to_failure FLOAT,
    maintenance_recommendations JSONB,
    confidence_score FLOAT,
    quantum_enhancement FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Autonomous system decisions
CREATE TABLE IF NOT EXISTS applications.autonomous_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES qmann.models(id),
    agent_id VARCHAR(255) NOT NULL,
    mission_id VARCHAR(255),
    agent_state JSONB NOT NULL,
    decision_type VARCHAR(100) NOT NULL,
    action_taken JSONB NOT NULL,
    coordination_data JSONB,
    quantum_coherence FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- ============================================================================
-- MONITORING TABLES
-- ============================================================================

-- System metrics
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit logs
CREATE TABLE IF NOT EXISTS monitoring.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Training metrics indexes
CREATE INDEX IF NOT EXISTS idx_training_metrics_session_epoch ON qmann.training_metrics(session_id, epoch);
CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp ON qmann.training_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_training_metrics_name ON qmann.training_metrics(metric_name);

-- Quantum jobs indexes
CREATE INDEX IF NOT EXISTS idx_quantum_jobs_status ON quantum.jobs(status);
CREATE INDEX IF NOT EXISTS idx_quantum_jobs_backend ON quantum.jobs(backend_id);
CREATE INDEX IF NOT EXISTS idx_quantum_jobs_submitted ON quantum.jobs(submitted_at);

-- Application predictions indexes
CREATE INDEX IF NOT EXISTS idx_healthcare_patient ON applications.healthcare_predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_healthcare_timestamp ON applications.healthcare_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_industrial_equipment ON applications.industrial_predictions(equipment_id);
CREATE INDEX IF NOT EXISTS idx_industrial_timestamp ON applications.industrial_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent ON applications.autonomous_decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_timestamp ON applications.autonomous_decisions(timestamp);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON monitoring.system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON monitoring.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON monitoring.audit_logs(user_id);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- ============================================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON qmann.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON classical.datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default quantum backends
INSERT INTO quantum.backends (name, provider, backend_type, num_qubits, quantum_volume, gate_error_rate, readout_error_rate, coherence_time_t1, coherence_time_t2, configuration) VALUES
('qasm_simulator', 'Qiskit Aer', 'simulator', 32, NULL, 0.0, 0.0, NULL, NULL, '{"simulator": true, "max_qubits": 32}'),
('statevector_simulator', 'Qiskit Aer', 'simulator', 20, NULL, 0.0, 0.0, NULL, NULL, '{"simulator": true, "max_qubits": 20, "method": "statevector"}'),
('ibm_brisbane', 'IBM Quantum', 'hardware', 127, 64, 0.001, 0.02, 100e-6, 50e-6, '{"hardware": true, "location": "IBM Quantum Network"}'),
('ibm_kyoto', 'IBM Quantum', 'hardware', 127, 64, 0.001, 0.02, 100e-6, 50e-6, '{"hardware": true, "location": "IBM Quantum Network"}')
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA qmann TO qmann_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA quantum TO qmann_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA classical TO qmann_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA applications TO qmann_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO qmann_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA qmann TO qmann_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA quantum TO qmann_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA classical TO qmann_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA applications TO qmann_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO qmann_user;
