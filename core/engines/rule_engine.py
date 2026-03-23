# core/engines/rule_engine.py
"""Rule-based fraud detection engine using deterministic government eligibility criteria."""
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from datetime import datetime
from collections import defaultdict

from core.engines.base import BaseDetectionEngine
from core.models.beneficiary import FraudIndicator

class RuleBasedEngine(BaseDetectionEngine):
    """
    Deterministic rule engine for fraud detection.
    Checks: Duplicate IDs, Income eligibility, Shared accounts, Address clustering.
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", weight: float = 0.30):
        super().__init__(name="rule", weight=weight, db_path=db_path)
        
        # Scheme income thresholds (annual income in INR)
        self.scheme_thresholds = {
            'PDS': 300000,      # Below Poverty Line approx
            'PAHAL': float('inf'),  # Universal but suspicious if multiple connections
            'PM_KISAN': 2000000,    # 20L limit for farmers (relaxed)
            'PENSION': 200000,      # Low income seniors
            'SCHOLARSHIP': 800000   # Family income limit
        }
        
        # Suspicious thresholds
        self.max_ben_per_address = 5
        self.max_ben_per_phone = 3
        self.high_income_threshold = 1000000  # 10 Lakhs
        
        # Pre-compute duplicate statistics for scoring calibration
        self.duplicate_stats = self._precompute_duplicate_stats()
        
    def train(self, data: Optional[Any] = None) -> None:
        """Pre-compute statistics for rule calibration."""
        self.duplicate_stats = self._precompute_duplicate_stats()
        self.is_trained = True
        
    def _precompute_duplicate_stats(self) -> Dict:
        """Pre-calculate duplicate counts for efficient scoring."""
        stats = {
            'aadhaar_counts': {},
            'bank_counts': {},
            'phone_counts': {},
            'address_counts': {}
        }
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Aadhaar duplicates
            cursor.execute("""
                SELECT aadhaar_hash, COUNT(*) as cnt 
                FROM beneficiaries 
                GROUP BY aadhaar_hash 
                HAVING cnt > 1
            """)
            stats['aadhaar_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Bank account duplicates
            cursor.execute("""
                SELECT bank_hash, COUNT(*) as cnt 
                FROM beneficiaries 
                GROUP BY bank_hash 
                HAVING cnt > 1
            """)
            stats['bank_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Phone duplicates
            cursor.execute("""
                SELECT phone_hash, COUNT(*) as cnt 
                FROM beneficiaries 
                WHERE phone_hash IS NOT NULL
                GROUP BY phone_hash 
                HAVING cnt > 1
            """)
            stats['phone_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Address clustering (only high counts to save memory)
            cursor.execute("""
                SELECT address, COUNT(*) as cnt 
                FROM beneficiaries 
                GROUP BY address 
                HAVING cnt > 3
            """)
            stats['address_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
            
        return stats
    
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """Analyze single beneficiary against all rules."""
        data = self.get_beneficiary_data(beneficiary_id)
        if not data:
            return FraudIndicator(
                engine=self.name,
                score=0,
                severity="low",
                description="Beneficiary not found in database"
            )
        
        violations = []
        score = 0
        
        # Check 1: Duplicate Aadhaar (Identity Fraud)
        dup_aadhaar = self._check_duplicate_aadhaar(data)
        if dup_aadhaar:
            violations.append(dup_aadhaar)
            score = max(score, dup_aadhaar['score_contribution'])
        
        # Check 2: Income Eligibility
        income_check = self._check_income_eligibility(data)
        if income_check:
            violations.append(income_check)
            score = max(score, income_check['score_contribution'])
        
        # Check 3: Shared Bank Account (Fraud Ring)
        bank_check = self._check_shared_bank(data)
        if bank_check:
            violations.append(bank_check)
            score = max(score, bank_check['score_contribution'])
        
        # Check 4: Address Clustering
        addr_check = self._check_address_clustering(data)
        if addr_check:
            violations.append(addr_check)
            score = max(score, addr_check['score_contribution'])
        
        # Check 5: Shared Phone Number
        phone_check = self._check_shared_phone(data)
        if phone_check:
            violations.append(phone_check)
            score = max(score, phone_check['score_contribution'])
        
        # Check 6: High Income + Subsidy Mismatch
        mismatch_check = self._check_income_subsidy_mismatch(data)
        if mismatch_check:
            violations.append(mismatch_check)
            score = max(score, mismatch_check['score_contribution'])
        
        # Check 7: Account Status
        status_check = self._check_account_status(data)
        if status_check:
            violations.append(status_check)
        
        # Create description from violations
        if violations:
            primary = max(violations, key=lambda x: x['score_contribution'])
            description = f"Rule Violations ({len(violations)}): {primary['reason']}"
            details = {
                'violation_count': len(violations),
                'violations': violations,
                'primary_violation': primary['type']
            }
        else:
            description = "No rule violations detected"
            details = {'violations': []}
        
        return FraudIndicator(
            engine=self.name,
            score=min(100, score),
            severity=self._score_to_severity(score),
            description=description,
            details=details
        )
    
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """Batch analysis for efficiency."""
        results = []
        
        # Get all data in one query
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(beneficiary_ids))
            cursor.execute(f"""
                SELECT b.*, 
                       COUNT(t.transaction_id) as txn_count,
                       GROUP_CONCAT(DISTINCT t.scheme_type) as schemes
                FROM beneficiaries b
                LEFT JOIN transactions t ON b.beneficiary_id = t.beneficiary_id
                WHERE b.beneficiary_id IN ({placeholders})
                GROUP BY b.beneficiary_id
            """, beneficiary_ids)
            
            rows = {row['beneficiary_id']: dict(row) for row in cursor.fetchall()}
        
        # Analyze each
        for bid in beneficiary_ids:
            if bid in rows:
                data = rows[bid]
                # Calculate violations
                violations = self._calculate_all_violations(data)
                score = max([v['score_contribution'] for v in violations], default=0)
                
                indicator = FraudIndicator(
                    engine=self.name,
                    score=min(100, score),
                    severity=self._score_to_severity(score),
                    description=f"Rule Violations: {len(violations)}" if violations else "Clean",
                    details={'violations': violations}
                )
            else:
                indicator = FraudIndicator(
                    engine=self.name,
                    score=0,
                    severity="low",
                    description="Not found"
                )
            results.append(indicator)
            
        return results
    
    def _calculate_all_violations(self, data: Dict) -> List[Dict]:
        """Calculate all violations for a data row."""
        violations = []
        
        checks = [
            self._check_duplicate_aadhaar(data),
            self._check_income_eligibility(data),
            self._check_shared_bank(data),
            self._check_address_clustering(data),
            self._check_shared_phone(data),
            self._check_income_subsidy_mismatch(data),
            self._check_account_status(data)
        ]
        
        return [c for c in checks if c]
    
    def _check_duplicate_aadhaar(self, data: Dict) -> Optional[Dict]:
        """Check if Aadhaar is shared by multiple beneficiaries."""
        aadhaar_hash = data['aadhaar_hash']
        
        if aadhaar_hash in self.duplicate_stats['aadhaar_counts']:
            count = self.duplicate_stats['aadhaar_counts'][aadhaar_hash]
            
            # Score based on severity (2 people = 70, 3+ = 90-100)
            if count == 2:
                score = 70
            elif count == 3:
                score = 85
            else:
                score = 95
            
            return {
                'type': 'duplicate_aadhaar',
                'reason': f"Aadhaar shared by {count} beneficiaries (Identity Fraud)",
                'score_contribution': score,
                'count': count,
                'feature': 'aadhaar_uniqueness',
                'evidence': f"Found {count-1} other registrations with same Aadhaar"
            }
        return None
    
    # Check 2: Income Eligibility (now uses beneficiary.scheme_type)
def _check_income_eligibility(self, data: Dict) -> Optional[Dict]:
    """Check if income exceeds scheme eligibility."""
    income = data['annual_income']
    scheme = data.get('scheme_type', 'PDS')  # Get scheme from beneficiary table
    
    threshold = self.scheme_thresholds.get(scheme, float('inf'))
    
    if income > threshold:
        excess = income - threshold
        severity = "critical" if excess > 500000 else "high" if excess > 100000 else "medium"
        
        return {
            'type': 'income_ineligible',
            'reason': f"Income ₹{income:,.0f} exceeds {scheme} limit (₹{threshold:,.0f})",
            'score_contribution': 85 if excess > 500000 else 70 if excess > 100000 else 60,
            'scheme': scheme,
            'feature': 'income_eligibility'
        }
    return None
    
    def _check_shared_bank(self, data: Dict) -> Optional[Dict]:
        """Check if bank account is shared (fraud ring indicator)."""
        bank_hash = data['bank_hash']
        
        if bank_hash in self.duplicate_stats['bank_counts']:
            count = self.duplicate_stats['bank_counts'][bank_hash]
            
            # Get details of other beneficiaries
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT beneficiary_id, name 
                    FROM beneficiaries 
                    WHERE bank_hash = ? AND beneficiary_id != ?
                """, (bank_hash, data['beneficiary_id']))
                others = cursor.fetchall()
            
            # High severity for shared bank accounts
            if count >= 5:
                score = 95
            elif count >= 3:
                score = 85
            else:
                score = 75
            
            return {
                'type': 'shared_bank_account',
                'reason': f"Bank account shared with {count-1} other beneficiary(ies) (Fraud Ring)",
                'score_contribution': score,
                'count': count,
                'linked_beneficiaries': [row[0] for row in others],
                'feature': 'bank_uniqueness',
                'evidence': f"Shared account with: {', '.join([row[1] for row in others[:3]])}"
            }
        return None
    
    def _check_address_clustering(self, data: Dict) -> Optional[Dict]:
        """Check if address has suspicious number of beneficiaries."""
        address = data['address']
        
        if address in self.duplicate_stats['address_counts']:
            count = self.duplicate_stats['address_counts'][address]
            
            if count > self.max_ben_per_address:
                # Score increases with cluster size
                excess = count - self.max_ben_per_address
                score = min(85, 60 + (excess * 5))  # +5 per excess person, max 85
                
                return {
                    'type': 'address_clustering',
                    'reason': f"Suspicious address cluster: {count} beneficiaries at same address",
                    'score_contribution': score,
                    'count': count,
                    'threshold': self.max_ben_per_address,
                    'feature': 'address_density',
                    'evidence': f"{count} people registered at: {address[:50]}..."
                }
        return None
    
    def _check_shared_phone(self, data: Dict) -> Optional[Dict]:
        """Check if phone number is shared."""
        phone_hash = data.get('phone_hash')
        if not phone_hash:
            return None
        
        if phone_hash in self.duplicate_stats['phone_counts']:
            count = self.duplicate_stats['phone_counts'][phone_hash]
            
            if count > self.max_ben_per_phone:
                score = min(70, 50 + (count - self.max_ben_per_phone) * 10)
                
                return {
                    'type': 'shared_phone',
                    'reason': f"Phone number shared by {count} beneficiaries",
                    'score_contribution': score,
                    'count': count,
                    'feature': 'phone_uniqueness',
                    'evidence': f"Phone used by {count} different people"
                }
        return None
    
    def _check_income_subsidy_mismatch(self, data: Dict) -> Optional[Dict]:
        """Check high income individuals claiming subsidies."""
        income = data['annual_income']
        
        if income > self.high_income_threshold:  # >10L
            # Check if claiming low-income schemes
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT SUM(amount) FROM transactions 
                    WHERE beneficiary_id = ? AND status = 'success'
                    AND scheme_type IN ('PDS', 'PENSION')
                """, (data['beneficiary_id'],))
                subsidy_amount = cursor.fetchone()[0] or 0
            
            if subsidy_amount > 0:
                return {
                    'type': 'high_income_subsidy',
                    'reason': f"High income (₹{income:,.0f}) claiming subsidies",
                    'score_contribution': 80,
                    'income': income,
                    'subsidy_claimed': subsidy_amount,
                    'feature': 'income_subsidy_mismatch',
                    'evidence': f"Income ₹{income:,.0f} but claimed ₹{subsidy_amount:,.0f} in welfare"
                }
        return None
    
    def _check_account_status(self, data: Dict) -> Optional[Dict]:
        """Check if account is already flagged."""
        status = data.get('status', 'active')
        
        if status in ['suspended', 'blocked', 'deceased']:
            severity_map = {
                'suspended': 70,
                'blocked': 90,
                'deceased': 100  # Ghost beneficiary - critical
            }
            
            return {
                'type': 'suspicious_status',
                'reason': f"Account status: {status}",
                'score_contribution': severity_map.get(status, 50),
                'status': status,
                'feature': 'account_status',
                'evidence': f"Status is {status}"
            }
        return None
    
    def _score_to_severity(self, score: float) -> str:
        """Convert score to severity level."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule triggers."""
        stats = {
            'duplicate_aadhaars_found': len(self.duplicate_stats['aadhaar_counts']),
            'shared_bank_accounts': len(self.duplicate_stats['bank_counts']),
            'shared_phones': len(self.duplicate_stats['phone_counts']),
            'address_clusters': len(self.duplicate_stats['address_counts'])
        }
        return stats

# Test function
if __name__ == "__main__":
    engine = RuleBasedEngine()
    print(f"✓ Rule Engine initialized")
    print(f"Statistics: {engine.get_rule_statistics()}")
    
    # Test on first beneficiary
    with engine.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT beneficiary_id FROM beneficiaries LIMIT 1")
        test_id = cursor.fetchone()[0]
    
    result = engine.analyze(test_id)
    print(f"\nTest Analysis for {test_id}:")
    print(f"Score: {result.score}")
    print(f"Severity: {result.severity}")
    print(f"Description: {result.description}")
    if result.details and result.details.get('violations'):
        for v in result.details['violations'][:3]:
            print(f"  - {v['type']}: {v['reason']}")