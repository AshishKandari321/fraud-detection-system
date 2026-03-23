# core/engines/base.py
"""Abstract base class for all fraud detection engines."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import sqlite3
from datetime import datetime

from core.models.beneficiary import Beneficiary, FraudIndicator

class BaseDetectionEngine(ABC):
    """
    Abstract base class for fraud detection strategies.
    All engines (Rule, Velocity, ML, Anomaly, Graph) inherit from this.
    """
    
    def __init__(self, name: str, weight: float = 1.0, db_path: str = "data/processed/fraud_system.db"):
        """
        Initialize detection engine.
        
        Args:
            name: Engine identifier (e.g., 'rule_based', 'velocity')
            weight: Weight for hybrid scoring (0.0 to 1.0)
            db_path: Path to SQLite database
        """
        self.name = name
        self.weight = weight
        self.db_path = db_path
        self.is_trained = False
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'description': self.__doc__
        }
    
    def get_db_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    @abstractmethod
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """
        Analyze a single beneficiary and return fraud indicator.
        
        Args:
            beneficiary_id: Unique beneficiary identifier
            
        Returns:
            FraudIndicator with score (0-100) and details
        """
        pass
    
    @abstractmethod
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """
        Batch analysis for efficiency.
        
        Args:
            beneficiary_ids: List of beneficiary IDs
            
        Returns:
            List of FraudIndicators (same order as input)
        """
        pass
    
    @abstractmethod
    def train(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the engine if needed (for ML models).
        Rule-based engines can skip this or pre-compute statistics.
        
        Args:
            data: Training data (optional, can query DB instead)
        """
        pass
    
    def get_engine_score(self, indicators: List[FraudIndicator]) -> float:
        """
        Calculate weighted contribution to overall fraud score.
        
        Args:
            indicators: List of indicators from this engine
            
        Returns:
            Weighted average score (0-100)
        """
        if not indicators:
            return 50.0  # Neutral if no indicators
        
        avg_score = sum(ind.score for ind in indicators) / len(indicators)
        return min(100, max(0, avg_score * self.weight))
    
    def validate_beneficiary(self, beneficiary_id: str) -> bool:
        """Check if beneficiary exists in database."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM beneficiaries WHERE beneficiary_id = ?",
                    (beneficiary_id,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def get_beneficiary_data(self, beneficiary_id: str) -> Optional[Dict[str, Any]]:
        """Fetch complete beneficiary record from DB."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT b.*, 
                           COUNT(t.transaction_id) as txn_count,
                           SUM(CASE WHEN t.status = 'success' THEN t.amount ELSE 0 END) as total_claimed
                    FROM beneficiaries b
                    LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                    WHERE b.beneficiary_id = ?
                    GROUP BY b.beneficiary_id
                """, (beneficiary_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            print(f"Error fetching beneficiary {beneficiary_id}: {e}")
            return None
    
    def get_transaction_history(self, beneficiary_id: str, 
                               days: int = 365) -> List[Dict[str, Any]]:
        """Get recent transaction history."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM transactions 
                    WHERE beneficiary_id = ? 
                    AND transaction_date >= date('now', '-{} days')
                    ORDER BY transaction_date DESC
                """.format(days), (beneficiary_id,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching transactions for {beneficiary_id}: {e}")
            return []
    
    def pre_compute_statistics(self) -> Dict[str, Any]:
        """
        Pre-compute global statistics for scoring calibration.
        Called during initialization or training.
        """
        stats = {}
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Income statistics by scheme
                cursor.execute("""
                    SELECT scheme_type, 
                           AVG(amount) as avg_amount,
                           COUNT(*) as count
                    FROM transactions 
                    WHERE status = 'success'
                    GROUP BY scheme_type
                """)
                stats['scheme_stats'] = {row[0]: {'avg': row[1], 'count': row[2]} 
                                        for row in cursor.fetchall()}
                
                # Duplicate Aadhaar count (fraud signal)
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT aadhaar_hash 
                        FROM beneficiaries 
                        GROUP BY aadhaar_hash 
                        HAVING COUNT(*) > 1
                    )
                """)
                stats['duplicate_aadhaars'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            print(f"Error computing statistics: {e}")
            return {}
    
    def explain(self, indicator: FraudIndicator) -> str:
        """
        Generate human-readable explanation.
        Override in subclasses for specific explanations.
        """
        return f"{self.name} Engine: Score {indicator.score:.1f} - {indicator.description}"