# core/engines/ml_engine.py
"""Supervised ML engine using Random Forest and XGBoost."""
import sqlite3
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

from core.engines.base import BaseDetectionEngine
from core.models.beneficiary import FraudIndicator


class MLEngine(BaseDetectionEngine):
    """
    Machine Learning fraud detection using ensemble of Random Forest and XGBoost.
    Features: Income ratios, transaction patterns, demographic anomalies.
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", weight: float = 0.15):
        super().__init__(name="ml", weight=weight, db_path=db_path)
        
        self.models = {
            'random_forest': None,
            'xgboost': None
        }
        self.feature_columns = []
        self.scaler_params = {}
        self.model_path = "data/processed/ml_models.pkl"
        
    def train(self, data: Optional[pd.DataFrame] = None) -> None:
        """Train ML models on labeled data."""
        print("Training ML models...")
        
        # Load training data from database
        if data is None:
            data = self._load_training_data()
        
        if len(data) < 100:
            print("  ⚠️ Insufficient data for ML training")
            return
            
        # Prepare features
        X, y = self._prepare_features(data)
        
        if len(X) == 0:
            print("  ⚠️ No valid features extracted")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        print("  Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Train XGBoost
        print("  Training XGBoost...")
        xg_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=5,  # Handle class imbalance
            eval_metric='logloss',
            random_state=42
        )
        xg_model.fit(X_train, y_train)
        self.models['xgboost'] = xg_model
        
        # Evaluate
        rf_pred = rf.predict_proba(X_test)[:, 1]
        xg_pred = xg_model.predict_proba(X_test)[:, 1]
        ensemble_pred = (rf_pred + xg_pred) / 2
        
        auc = roc_auc_score(y_test, ensemble_pred)
        print(f"  ✓ Ensemble AUC: {auc:.3f}")
        
        # Store feature importance
        self.feature_columns = X.columns.tolist()
        
        # Save models
        self._save_models()
        self.is_trained = True
        print(f"  ✓ ML Engine trained on {len(X)} samples")
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load data with fraud labels derived from duplicate patterns."""
        with self.get_db_connection() as conn:
            # Create labels based on known fraud patterns in data
            query = """
                SELECT 
                    b.beneficiary_id,
                    b.annual_income,
                    b.family_size,
                    b.district,
                    b.state,
                    COUNT(t.transaction_id) as txn_count,
                    AVG(t.amount) as avg_amount,
                    SUM(CASE WHEN t.status = 'success' THEN t.amount ELSE 0 END) as total_claimed,
                    COUNT(DISTINCT t.agent_id) as unique_agents,
                    COUNT(DISTINCT t.scheme_type) as schemes_used,
                    MAX(CASE WHEN b.annual_income > 1000000 THEN 1 ELSE 0 END) as high_income_flag,
                    -- Create labels from duplicate patterns (fraud indicators)
                    CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM beneficiaries b2 
                            WHERE b2.aadhaar_hash = b.aadhaar_hash 
                            AND b2.beneficiary_id != b.beneficiary_id
                        ) THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM beneficiaries b2 
                            WHERE b2.bank_hash = b.bank_hash 
                            AND b2.beneficiary_id != b.beneficiary_id
                        ) THEN 1
                        WHEN b.annual_income > 1000000 AND EXISTS (
                            SELECT 1 FROM transactions t2 
                            WHERE t2.beneficiary_id = b.beneficiary_id 
                            AND t2.scheme_type IN ('PDS', 'PENSION')
                        ) THEN 1
                        ELSE 0 
                    END as is_fraud
                FROM beneficiaries b
                LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                GROUP BY b.beneficiary_id
            """
            return pd.read_sql_query(query, conn)
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Engineer features for ML."""
        # Handle missing
        df = df.fillna(0)
        
        # Feature engineering
        features = pd.DataFrame()
        features['income'] = pd.to_numeric(df['annual_income'], errors='coerce').fillna(0)
        features['income_log'] = np.log1p(features['income'])
        features['txn_count'] = df['txn_count'].fillna(0)
        features['avg_amount'] = df['avg_amount'].fillna(0)
        features['total_claimed'] = df['total_claimed'].fillna(0)
        features['unique_agents'] = df['unique_agents'].fillna(0)
        features['schemes_used'] = df['schemes_used'].fillna(0)
        features['claim_income_ratio'] = features['total_claimed'] / (features['income'] + 1)
        features['family_size'] = pd.to_numeric(df['family_size'], errors='coerce').fillna(4)
        
        # Normalize
        for col in features.columns:
            if col not in self.scaler_params:
                self.scaler_params[col] = {
                    'mean': features[col].mean(),
                    'std': features[col].std() + 1e-8
                }
            features[col] = (features[col] - self.scaler_params[col]['mean']) / self.scaler_params[col]['std']
        
        target = df['is_fraud'].fillna(0)
        
        return features, target
    
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """Predict fraud probability using ML ensemble."""
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self._load_models()
            else:
                # Train on the fly if no saved model
                self.train()
        
        # Load beneficiary data
        data = self._load_single_beneficiary(beneficiary_id)
        if data is None:
            return FraudIndicator(
                engine=self.name,
                score=50,
                severity="medium",
                description="ML: Beneficiary data not found"
            )
        
        # Prepare features
        X = self._prepare_single_features(data)
        
        # Get predictions from both models
        rf_prob = self.models['random_forest'].predict_proba(X)[0][1]
        xg_prob = self.models['xgboost'].predict_proba(X)[0][1]
        
        # Ensemble average
        ensemble_prob = (rf_prob + xg_prob) / 2
        score = ensemble_prob * 100  # Convert to 0-100
        
        # Feature importance for this prediction
        feature_imp = self._get_feature_importance(X)
        
        severity = "high" if score > 70 else "medium" if score > 40 else "low"
        
        return FraudIndicator(
            engine=self.name,
            score=round(score, 2),
            severity=severity,
            description=f"ML Ensemble: {score:.1f}% fraud probability",
            details={
                'rf_probability': round(rf_prob * 100, 2),
                'xg_probability': round(xg_prob * 100, 2),
                'feature_importance': feature_imp,
                'top_features': sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        )
    
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """Batch prediction."""
        return [self.analyze(bid) for bid in beneficiary_ids]
    
    def _load_single_beneficiary(self, beneficiary_id: str) -> Optional[pd.DataFrame]:
        """Load data for single beneficiary."""
        with self.get_db_connection() as conn:
            query = """
                SELECT 
                    b.beneficiary_id,
                    b.annual_income,
                    b.family_size,
                    COUNT(t.transaction_id) as txn_count,
                    AVG(t.amount) as avg_amount,
                    SUM(CASE WHEN t.status = 'success' THEN t.amount ELSE 0 END) as total_claimed,
                    COUNT(DISTINCT t.agent_id) as unique_agents,
                    COUNT(DISTINCT t.scheme_type) as schemes_used
                FROM beneficiaries b
                LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                WHERE b.beneficiary_id = ?
                GROUP BY b.beneficiary_id
            """
            df = pd.read_sql_query(query, conn, params=(beneficiary_id,))
            return df if len(df) > 0 else None
    
    def _prepare_single_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for single prediction."""
        df = df.fillna(0)
        
        features = pd.DataFrame()
        features['income'] = pd.to_numeric(df['annual_income'], errors='coerce').fillna(0)
        features['income_log'] = np.log1p(features['income'])
        features['txn_count'] = df['txn_count'].fillna(0)
        features['avg_amount'] = df['avg_amount'].fillna(0)
        features['total_claimed'] = df['total_claimed'].fillna(0)
        features['unique_agents'] = df['unique_agents'].fillna(0)
        features['schemes_used'] = df['schemes_used'].fillna(0)
        features['claim_income_ratio'] = features['total_claimed'] / (features['income'] + 1)
        features['family_size'] = pd.to_numeric(df['family_size'], errors='coerce').fillna(4)
        
        # Ensure column order matches training
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_columns]
        
        # Normalize
        for col in features.columns:
            if col in self.scaler_params:
                features[col] = (features[col] - self.scaler_params[col]['mean']) / self.scaler_params[col]['std']
        
        return features
    
    def _get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance for explanation."""
        if not self.is_trained:
            return {}
        
        # Use Random Forest feature importance as proxy
        importances = self.models['random_forest'].feature_importances_
        return dict(zip(self.feature_columns, importances))
    
    def _save_models(self):
        """Save trained models."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_columns': self.feature_columns,
                'scaler_params': self.scaler_params
            }, f)
    
    def _load_models(self):
        """Load saved models."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.feature_columns = data['feature_columns']
            self.scaler_params = data['scaler_params']
            self.is_trained = True


if __name__ == "__main__":
    engine = MLEngine()
    engine.train()
    
    # Test
    with engine.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT beneficiary_id FROM beneficiaries LIMIT 3")
        test_ids = [r[0] for r in cursor.fetchall()]
        
    for bid in test_ids:
        result = engine.analyze(bid)
        print(f"{bid}: {result.score:.1f} - {result.description}")