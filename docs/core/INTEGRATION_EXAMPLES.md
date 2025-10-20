# 🔗 QMANN v2.0 - Integration Examples

**Version**: 2.0.0  
**Last Updated**: 2025-10-20  
**Status**: ✅ Complete

---

## 📋 Table of Contents

1. [Healthcare Integration](#healthcare-integration)
2. [Finance Integration](#finance-integration)
3. [Drug Discovery Integration](#drug-discovery-integration)
4. [Industrial Integration](#industrial-integration)
5. [REST API Integration](#rest-api-integration)
6. [Database Integration](#database-integration)
7. [Cloud Integration](#cloud-integration)

---

## 🏥 Healthcare Integration

### Example 1: Postoperative Complication Prediction

```python
from qmann.applications.healthcare import HealthcarePredictor
from qmann.core.config import get_config_for_application
import pandas as pd

# Load patient data
patient_data = pd.read_csv('patient_data.csv')

# Get healthcare-optimized config
config = get_config_for_application('healthcare')

# Initialize predictor
predictor = HealthcarePredictor(config=config)

# Make predictions
predictions = predictor.predict(patient_data)

# Extract results
for idx, pred in enumerate(predictions):
    print(f"Patient {idx}:")
    print(f"  Complication Risk: {pred['risk_score']:.4f}")
    print(f"  Sensitivity: {pred['sensitivity']:.4f}")
    print(f"  Specificity: {pred['specificity']:.4f}")
    print(f"  Recommended Action: {pred['action']}")

# Get risk factors
risk_factors = predictor.get_risk_factors()
print(f"\nTop Risk Factors: {risk_factors[:5]}")

# Explain specific prediction
explanation = predictor.explain_prediction(patient_data.iloc[0])
print(f"\nPrediction Explanation: {explanation}")
```

### Example 2: Real-time Monitoring

```python
from qmann.applications.healthcare import HealthcarePredictor
import asyncio
import json

class HealthcareMonitor:
    def __init__(self, config):
        self.predictor = HealthcarePredictor(config=config)
        self.alert_threshold = 0.7
    
    async def monitor_patient(self, patient_id, vital_signs):
        """Monitor patient in real-time"""
        # Prepare features
        features = self._prepare_features(vital_signs)
        
        # Get prediction
        prediction = self.predictor.predict(features)
        
        # Check alert condition
        if prediction['risk_score'] > self.alert_threshold:
            await self._send_alert(patient_id, prediction)
        
        return prediction
    
    async def _send_alert(self, patient_id, prediction):
        """Send alert to medical staff"""
        alert = {
            'patient_id': patient_id,
            'risk_score': prediction['risk_score'],
            'timestamp': datetime.now().isoformat(),
            'action': 'IMMEDIATE_REVIEW_REQUIRED'
        }
        # Send to alert system
        print(f"ALERT: {json.dumps(alert)}")

# Usage
config = get_config_for_application('healthcare')
monitor = HealthcareMonitor(config)

# Simulate real-time monitoring
vital_signs = {
    'heart_rate': 95,
    'blood_pressure': 140,
    'temperature': 37.5,
    'oxygen_saturation': 95
}

result = asyncio.run(monitor.monitor_patient('P001', vital_signs))
```

---

## 💰 Finance Integration

### Example 1: Portfolio Optimization

```python
from qmann.applications.finance import FinancePredictor
from qmann.core.config import get_config_for_application
import numpy as np

# Get finance-optimized config
config = get_config_for_application('finance')

# Initialize predictor
predictor = FinancePredictor(config=config)

# Asset returns (historical data)
asset_returns = np.random.randn(100, 5)  # 100 days, 5 assets

# Optimize portfolio
portfolio = predictor.optimize_portfolio(
    assets=asset_returns,
    constraints={
        'max_risk': 0.15,
        'min_return': 0.05,
        'max_concentration': 0.3
    }
)

print(f"Optimal Portfolio:")
print(f"  Weights: {portfolio['weights']}")
print(f"  Expected Return: {portfolio['expected_return']:.4f}")
print(f"  Risk (Std Dev): {portfolio['risk']:.4f}")
print(f"  Sharpe Ratio: {portfolio['sharpe_ratio']:.4f}")

# Calculate Value at Risk
var_95 = predictor.calculate_var(asset_returns, confidence=0.95)
print(f"\nValue at Risk (95%): {var_95:.4f}")

# Predict future returns
future_features = np.random.randn(1, 10)
predicted_returns = predictor.predict_returns(future_features)
print(f"Predicted Returns: {predicted_returns}")
```

### Example 2: Risk Analysis Dashboard

```python
from qmann.applications.finance import FinancePredictor
from fastapi import FastAPI
import json

app = FastAPI()
predictor = FinancePredictor(config=get_config_for_application('finance'))

@app.get("/portfolio/risk")
async def get_portfolio_risk(portfolio_id: str):
    """Get portfolio risk metrics"""
    portfolio = load_portfolio(portfolio_id)
    
    # Calculate metrics
    var = predictor.calculate_var(portfolio['returns'], confidence=0.95)
    cvar = predictor.calculate_cvar(portfolio['returns'], confidence=0.95)
    
    return {
        'portfolio_id': portfolio_id,
        'var_95': var,
        'cvar_95': cvar,
        'sharpe_ratio': portfolio['sharpe_ratio'],
        'max_drawdown': portfolio['max_drawdown']
    }

@app.post("/portfolio/optimize")
async def optimize_portfolio(portfolio_data: dict):
    """Optimize portfolio allocation"""
    optimized = predictor.optimize_portfolio(
        assets=portfolio_data['returns'],
        constraints=portfolio_data['constraints']
    )
    
    return {
        'weights': optimized['weights'],
        'expected_return': optimized['expected_return'],
        'risk': optimized['risk'],
        'sharpe_ratio': optimized['sharpe_ratio']
    }
```

---

## 🧪 Drug Discovery Integration

### Example 1: Molecular Property Prediction

```python
from qmann.applications.drug_discovery import DrugDiscoveryPredictor
from qmann.core.config import get_config_for_application
import pandas as pd

# Get drug discovery config
config = get_config_for_application('drug_discovery')

# Initialize predictor
predictor = DrugDiscoveryPredictor(config=config)

# Load molecular data
molecules = pd.read_csv('molecules.csv')

# Predict properties
properties = predictor.predict_properties(molecules)

# Filter drug-like candidates
drug_like = properties[properties['drug_likeness'] > 0.8]
print(f"Drug-like candidates: {len(drug_like)}/{len(properties)}")

# Rank candidates
ranked = predictor.rank_candidates(drug_like)
print(f"\nTop 5 Candidates:")
for i, candidate in enumerate(ranked[:5], 1):
    print(f"{i}. {candidate['name']}")
    print(f"   Drug-likeness: {candidate['drug_likeness']:.4f}")
    print(f"   Toxicity Risk: {candidate['toxicity_risk']:.4f}")
    print(f"   Synthesis Score: {candidate['synthesis_score']:.4f}")
```

### Example 2: Compound Screening Pipeline

```python
from qmann.applications.drug_discovery import DrugDiscoveryPredictor
from qmann.utils.benchmarks import Benchmarks

class CompoundScreeningPipeline:
    def __init__(self, config):
        self.predictor = DrugDiscoveryPredictor(config=config)
        self.benchmarks = Benchmarks(config=config)
    
    def screen_compounds(self, compound_library):
        """Screen large compound library"""
        results = []
        
        for compound in compound_library:
            # Predict properties
            properties = self.predictor.predict_properties(compound)
            
            # Apply filters
            if self._passes_filters(properties):
                results.append({
                    'compound': compound,
                    'properties': properties,
                    'score': self._calculate_score(properties)
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _passes_filters(self, properties):
        """Check if compound passes filters"""
        return (
            properties['drug_likeness'] > 0.7 and
            properties['toxicity_risk'] < 0.3 and
            properties['synthesis_score'] > 0.5
        )
    
    def _calculate_score(self, properties):
        """Calculate compound score"""
        return (
            0.4 * properties['drug_likeness'] +
            0.3 * (1 - properties['toxicity_risk']) +
            0.3 * properties['synthesis_score']
        )

# Usage
config = get_config_for_application('drug_discovery')
pipeline = CompoundScreeningPipeline(config)

# Screen compounds
results = pipeline.screen_compounds(compound_library)
print(f"Screened {len(results)} promising compounds")
```

---

## 🏭 Industrial Integration

### Example 1: Predictive Maintenance

```python
from qmann.applications.industrial import IndustrialMaintenance
from qmann.core.config import get_config_for_application
import numpy as np

# Get industrial config
config = get_config_for_application('industrial')

# Initialize maintenance predictor
maintenance = IndustrialMaintenance(config=config)

# Sensor data (temperature, vibration, pressure, etc.)
sensor_data = np.random.randn(1000, 10)

# Predict failures
predictions = maintenance.predict_failures(sensor_data)

print(f"Equipment Status:")
for equipment_id, pred in enumerate(predictions):
    print(f"Equipment {equipment_id}:")
    print(f"  Failure Probability: {pred['failure_prob']:.4f}")
    print(f"  Time to Failure: {pred['ttf']:.2f} hours")
    print(f"  Recommended Action: {pred['action']}")

# Get anomalies
anomalies = maintenance.detect_anomalies(sensor_data)
print(f"\nDetected {len(anomalies)} anomalies")
```

### Example 2: Real-time Monitoring System

```python
from qmann.applications.industrial import IndustrialMaintenance
import asyncio
from datetime import datetime

class IndustrialMonitoringSystem:
    def __init__(self, config):
        self.maintenance = IndustrialMaintenance(config=config)
        self.alert_threshold = 0.8
    
    async def monitor_equipment(self, equipment_id, sensor_stream):
        """Monitor equipment in real-time"""
        async for sensor_data in sensor_stream:
            # Get prediction
            prediction = self.maintenance.predict_failures(sensor_data)
            
            # Check alert condition
            if prediction['failure_prob'] > self.alert_threshold:
                await self._schedule_maintenance(equipment_id, prediction)
            
            # Log metrics
            self._log_metrics(equipment_id, prediction)
    
    async def _schedule_maintenance(self, equipment_id, prediction):
        """Schedule maintenance"""
        maintenance_order = {
            'equipment_id': equipment_id,
            'scheduled_time': datetime.now(),
            'priority': 'HIGH' if prediction['failure_prob'] > 0.9 else 'MEDIUM',
            'estimated_ttf': prediction['ttf']
        }
        print(f"Maintenance scheduled: {maintenance_order}")
    
    def _log_metrics(self, equipment_id, prediction):
        """Log monitoring metrics"""
        print(f"[{datetime.now()}] Equipment {equipment_id}: "
              f"Failure Prob={prediction['failure_prob']:.4f}")
```

---

## 🌐 REST API Integration

### Example: FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qmann.applications.healthcare import HealthcarePredictor
from qmann.core.config import get_config_for_application

app = FastAPI(title="QMANN API", version="2.0.0")

# Initialize predictor
config = get_config_for_application('healthcare')
predictor = HealthcarePredictor(config=config)

class PatientData(BaseModel):
    age: int
    heart_rate: int
    blood_pressure: str
    temperature: float

@app.post("/predict/complication")
async def predict_complication(patient: PatientData):
    """Predict postoperative complications"""
    try:
        prediction = predictor.predict(patient.dict())
        return {
            'risk_score': prediction['risk_score'],
            'sensitivity': prediction['sensitivity'],
            'specificity': prediction['specificity'],
            'action': prediction['action']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'version': '2.0.0'}

# Run: uvicorn main:app --reload
```

---

## 💾 Database Integration

### Example: SQLAlchemy Integration

```python
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from qmann.applications.healthcare import HealthcarePredictor

Base = declarative_base()

class PredictionResult(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String)
    risk_score = Column(Float)
    sensitivity = Column(Float)
    specificity = Column(Float)
    timestamp = Column(String)

# Create database
engine = create_engine('postgresql://user:password@localhost/qmann')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Use with predictor
predictor = HealthcarePredictor(config=config)
session = Session()

# Make prediction and save
prediction = predictor.predict(patient_data)
result = PredictionResult(
    patient_id='P001',
    risk_score=prediction['risk_score'],
    sensitivity=prediction['sensitivity'],
    specificity=prediction['specificity'],
    timestamp=datetime.now().isoformat()
)
session.add(result)
session.commit()
```

---

## ☁️ Cloud Integration

### Example: AWS Lambda Integration

```python
import json
import boto3
from qmann.applications.healthcare import HealthcarePredictor

predictor = HealthcarePredictor(config=get_config_for_application('healthcare'))

def lambda_handler(event, context):
    """AWS Lambda handler"""
    try:
        # Parse input
        patient_data = json.loads(event['body'])
        
        # Make prediction
        prediction = predictor.predict(patient_data)
        
        # Save to DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('qmann-predictions')
        table.put_item(Item={
            'patient_id': patient_data['id'],
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'statusCode': 200,
            'body': json.dumps(prediction)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

---

## 📞 Integration Support

- **Integration Questions**: https://github.com/neuraparse/qmann-v2.0/discussions
- **API Documentation**: See `API_DOCUMENTATION.md`
- **Architecture Guide**: See `ARCHITECTURE_GUIDE.md`

---

**Status**: ✅ Complete  
**Last Updated**: 2025-10-20


