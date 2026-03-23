# core/engines/anomaly_engine.py
"""Unsupervised anomaly detection using Isolation Forest and K-Means clustering."""
import sqlite3
import pickle
from typing import List, Dict, Any, Optional
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.engines.base import BaseDetectionEngine
from core.models.beneficiary import FraudIndicator


class AnomalyEngine(BaseDetectionEngine):
    """
    Unsupervised anomaly detection for unknown fraud patterns.
    Uses Isolation Forest (outliers) + K-Means (cluster deviation).
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", weight: float = 0.10):
        super().__init__(name="anomaly", weight=weight, db_path=db_path)
        
        self.isolation_forest = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = "data/processed/anomaly_models.pkl"
        self.normal_cluster_centers = None
        
    def train(self, data: Optional[pd.DataFrame] = None) -> None:
        """Train anomaly detection models."""
        print("Training Anomaly Detection models...")
        
        if data is None:
            data = self._load_data()
        
        if len(data) < 100:
            print("  ⚠ Insufficient data")
            return
        
        # Prepare features
        X = self._prepare_features(data)
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        print("  Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.15,  # Assume 15% anomalies (matches fraud rate)
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        # Train K-Means (find normal clusters, flag deviations)
        print("  Training K-Means...")
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        self.normal_cluster_centers = self.kmeans.cluster_centers_
        
        self.is_trained = True
        self._save_models()
        print(f"  ✓ Anomaly Engine trained on {len(X)} samples")
    
    def _load_data(self) -> pd.DataFrame:
        """Load beneficiary data for training."""
        with self.get_db_connection() as conn:
            query = """
                SELECT 
                    b.beneficiary_id,
                    b.annual_income,
                    b.family_size,
                    COUNT(t.transaction_id) as txn_count,
                    AVG(t.amount) as avg_amount,
                    SUM(t.amount) as total_amount,
                    COUNT(DISTINCT t.agent_id) as unique_agents,
                    COUNT(DISTINCT t.scheme_type) as scheme_diversity
                FROM beneficiaries b
                LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                GROUP BY b.beneficiary_id
            """
            return pd.read_sql_query(query, conn)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for anomaly detection."""
        features = pd.DataFrame()
        
        # Financial features
        features['income'] = pd.to_numeric(df['annual_income'], errors='coerce').fillna(0)
        features['income_log'] = np.log1p(features['income'] + 1)
        
        # Transaction behavior
        features['txn_count'] = df['txn_count'].fillna(0)
        features['avg_amount'] = df['avg_amount'].fillna(0)
        features['total_amount'] = df['total_amount'].fillna(0)
        features['amount_per_txn'] = features['total_amount'] / (features['txn_count'] + 1)
        
        # Diversity metrics (high diversity can indicate fraud)
        features['unique_agents'] = df['unique_agents'].fillna(0)
        features['scheme_diversity'] = df['scheme_diversity'].fillna(0)
        features['family_size'] = pd.to_numeric(df['family_size'], errors='coerce').fillna(4)
        
        # Ratios
        features['agents_per_txn'] = features['unique_agents'] / (features['txn_count'] + 1)
        features['income_utilization'] = features['total_amount'] / (features['income'] + 1)
        
        # Handle infinities
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """Detect anomalies for single beneficiary."""
        if not self.is_trained:
            if os.path.exists(self.model_path):
                self._load_models()
            else:
                self.train()
        
        # Load data
        data = self._load_single(beneficiary_id)
        if data is None:
            return FraudIndicator(
                engine=self.name,
                score=50,
                severity="medium",
                description="Anomaly: Data not found"
            )
        
        # Prepare features
        X = self._prepare_single(data)
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest score (-1 to 1, where -1 is outlier)
        iso_score = self.isolation_forest.decision_function(X_scaled)[0]
        iso_anomaly_prob = (0.5 - iso_score) * 100  # Convert to 0-100 scale
        
        # K-Means distance from nearest cluster
        distances = self.kmeans.transform(X_scaled)[0]
        min_distance = min(distances)
        max_possible = np.max([np.linalg.norm(c) for c in self.normal_cluster_centers])
        kmeans_score = (min_distance / max_possible) * 100 if max_possible > 0 else 0
        
        # Combine scores (Isolation Forest has higher weight)
        combined_score = (iso_anomaly_prob * 0.7) + (kmeans_score * 0.3)
        
        # Severity
        if combined_score > 70:
            severity = "high"
        elif combined_score > 40:
            severity = "medium"
        else:
            severity = "low"
        
        # Determine anomaly type
        anomaly_type = self._classify_anomaly(X)
        
        return FraudIndicator(
            engine=self.name,
            score=round(combined_score, 2),
            severity=severity,
            description=f"Anomaly: {anomaly_type} (Score: {combined_score:.1f})",
            details={
                'isolation_forest_score': round(iso_anomaly_prob, 2),
                'cluster_distance_score': round(kmeans_score, 2),
                'anomaly_type': anomaly_type,
                'is_outlier': iso_score < 0,
                'cluster_id': int(self.kmeans.predict(X_scaled)[0])
            }
        )
    
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """Batch anomaly detection."""
        return [self.analyze(bid) for bid in beneficiary_ids]
    
    def _load_single(self, beneficiary_id: str) -> Optional[pd.DataFrame]:
        """Load single beneficiary data."""
        with self.get_db_connection() as conn:
            query = """
                SELECT 
                    b.beneficiary_id,
                    b.annual_income,
                    b.family_size,
                    COUNT(t.transaction_id) as txn_count,
                    AVG(t.amount) as avg_amount,
                    SUM(t.amount) as total_amount,
                    COUNT(DISTINCT t.agent_id) as unique_agents,
                    COUNT(DISTINCT t.scheme_type) as scheme_diversity
                FROM beneficiaries b
                LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                WHERE b.beneficiary_id = ?
                GROUP BY b.beneficiary_id
            """
            df = pd.read_sql_query(query, conn, params=(beneficiary_id,))
            return df if len(df) > 0 else None
    
    def _prepare_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for single prediction."""
        # Use same logic as training
        features = self._prepare_features(df)
        
        # Ensure column order
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        return features[self.feature_columns]
    
    def _classify_anomaly(self, X: pd.DataFrame) -> str:
        """Classify the type of anomaly."""
        descriptions = []
        
        if X['income_utilization'].iloc[0] > 0.5:
            descriptions.append("High subsidy ratio")
        if X['agents_per_txn'].iloc[0] > 0.8:
            descriptions.append("Excessive agent switching")
        if X['scheme_diversity'].iloc[0] > 3:
            descriptions.append("Multi-scheme exploitation")
        if X['txn_count'].iloc[0] > 10:
            descriptions.append("High transaction frequency")
        
        return ", ".join(descriptions) if descriptions else "Statistical outlier"
    
    def _save_models(self):
        """Save models."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'isolation_forest': self.isolation_forest,
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'cluster_centers': self.normal_cluster_centers
            }, f)
    
    def _load_models(self):
        """Load models."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.isolation_forest = data['isolation_forest']
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.normal_cluster_centers = data['cluster_centers']
            self.is_trained = True


if __name__ == "__main__":
    engine = AnomalyEngine()
    engine.train()
    
    # Test on random samples
    with engine.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT beneficiary_id FROM beneficiaries 
            ORDER BY RANDOM() LIMIT 5
        """)
        test_ids = [r[0] for r in cursor.fetchall()]
    
    print("\nTest Results:")
    for bid in test_ids:
        result = engine.analyze(bid)
        print(f"{bid}: {result.score:.1f} - {result.description}")