# core/engines/velocity_engine.py
"""Temporal fraud detection - analyzes transaction velocity, timing, and frequency patterns."""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict
import statistics

from core.engines.base import BaseDetectionEngine
from core.models.beneficiary import FraudIndicator

class VelocityEngine(BaseDetectionEngine):
    """
    Detects temporal fraud patterns:
    - Velocity attacks (multiple claims in short window)
    - Unusual timing (off-hours transactions)
    - Frequency anomalies (too many claims vs historical)
    - Geographic impossibility (distance/speed violations)
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", weight: float = 0.25):
        super().__init__(name="velocity", weight=weight, db_path=db_path)
        
        # Velocity thresholds
        self.daily_claim_limit = 2  # Max claims per day
        self.weekly_claim_limit = 5  # Max claims per week
        self.hourly_velocity_limit = 3  # Claims within 1 hour
        
        # Timing anomalies
        self.off_hours_start = 22  # 10 PM
        self.off_hours_end = 5     # 5 AM
        
        # Geographic speed limit (km/hour - impossible travel)
        self.max_travel_speed = 100  # km/h
        
        # Pre-compute baselines
        self.baselines = self._compute_temporal_baselines()
        
    def train(self, data: Optional[Any] = None):
        """Compute temporal baselines for anomaly detection."""
        self.baselines = self._compute_temporal_baselines()
        self.is_trained = True
        
    def _compute_temporal_baselines(self) -> Dict:
        """Compute normal transaction patterns for comparison."""
        baselines = {}
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Average claims per beneficiary per month
            cursor.execute("""
                SELECT 
                    beneficiary_id,
                    COUNT(*) as txn_count,
                    COUNT(DISTINCT date(transaction_date)) as active_days
                FROM transactions
                WHERE transaction_date >= date('now', '-90 days')
                GROUP BY beneficiary_id
            """)
            
            counts = [row['txn_count'] for row in cursor.fetchall()]
            if counts:
                baselines['avg_monthly_claims'] = statistics.mean(counts)
                baselines['std_monthly_claims'] = statistics.stdev(counts) if len(counts) > 1 else 1
            else:
                baselines['avg_monthly_claims'] = 4
                baselines['std_monthly_claims'] = 2
            
            # Peak transaction hours (normal behavior)
            cursor.execute("""
                SELECT strftime('%H', transaction_date) as hour, COUNT(*)
                FROM transactions
                GROUP BY hour
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """)
            baselines['peak_hours'] = [int(row[0]) for row in cursor.fetchall()]
            
        return baselines
    
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """Analyze temporal patterns for single beneficiary."""
        violations = []
        score = 0
        
        # Get all transactions for this beneficiary
        transactions = self.get_transaction_history(beneficiary_id, days=90)
        if len(transactions) < 2:
            return FraudIndicator(
                engine=self.name,
                score=0,
                severity="low",
                description="Insufficient transaction history for velocity analysis"
            )
        
        # Check 1: Velocity attacks (too many in short time)
        velocity_check = self._check_velocity_attack(transactions)
        if velocity_check:
            violations.append(velocity_check)
            score = max(score, velocity_check['score_contribution'])
        
        # Check 2: Off-hours transactions
        timing_check = self._check_off_hours(transactions)
        if timing_check:
            violations.append(timing_check)
            score = max(score, timing_check['score_contribution'])
        
        # Check 3: Frequency anomaly vs baseline
        freq_check = self._check_frequency_anomaly(beneficiary_id, transactions)
        if freq_check:
            violations.append(freq_check)
            score = max(score, freq_check['score_contribution'])
        
        # Check 4: Geographic impossibility (speed violations)
        geo_check = self._check_geographic_impossibility(transactions)
        if geo_check:
            violations.append(geo_check)
            score = max(score, geo_check['score_contribution'])
        
        # Check 5: Round-trip fraud (claim at A, then B, then A quickly)
        roundtrip_check = self._check_round_trip_fraud(transactions)
        if roundtrip_check:
            violations.append(roundtrip_check)
            score = max(score, roundtrip_check['score_contribution'])
        
        # Check 6: Holiday/Weekend anomaly (government offices closed)
        holiday_check = self._check_office_hours_violation(transactions)
        if holiday_check:
            violations.append(holiday_check)
            score = max(score, holiday_check['score_contribution'])
        
        if violations:
            primary = max(violations, key=lambda x: x['score_contribution'])
            description = f"Temporal anomalies ({len(violations)}): {primary['reason']}"
        else:
            description = "No velocity anomalies detected"
            
        return FraudIndicator(
            engine=self.name,
            score=min(100, score),
            severity=self._score_to_severity(score),
            description=description,
            details={
                'violation_count': len(violations),
                'violations': violations,
                'transaction_count': len(transactions),
                'analysis_period_days': 90
            }
        )
    
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """Batch analysis optimized for DB queries."""
        results = []
        
        # Bulk fetch all transactions
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(beneficiary_ids))
            cursor.execute(f"""
                SELECT * FROM transactions 
                WHERE beneficiary_id IN ({placeholders})
                AND transaction_date >= date('now', '-90 days')
                ORDER BY beneficiary_id, transaction_date
            """, beneficiary_ids)
            
            # Group by beneficiary
            txns_by_ben = defaultdict(list)
            for row in cursor.fetchall():
                txns_by_ben[row['beneficiary_id']].append(dict(row))
        
        # Analyze each
        for bid in beneficiary_ids:
            if bid in txns_by_ben:
                # Temporarily override for analysis
                transactions = txns_by_ben[bid]
                violations = []
                score = 0
                
                # Run all checks
                checks = [
                    self._check_velocity_attack(transactions),
                    self._check_off_hours(transactions),
                    self._check_frequency_anomaly(bid, transactions),
                    self._check_geographic_impossibility(transactions),
                    self._check_round_trip_fraud(transactions),
                    self._check_office_hours_violation(transactions)
                ]
                
                violations = [c for c in checks if c]
                score = max([v['score_contribution'] for v in violations], default=0)
                
                indicator = FraudIndicator(
                    engine=self.name,
                    score=min(100, score),
                    severity=self._score_to_severity(score),
                    description=f"Temporal: {len(violations)} anomalies" if violations else "Clean",
                    details={'violations': violations, 'txn_count': len(transactions)}
                )
            else:
                indicator = FraudIndicator(
                    engine=self.name,
                    score=0,
                    severity="low",
                    description="No recent transactions"
                )
            results.append(indicator)
            
        return results
    
    def _check_velocity_attack(self, transactions: List[Dict]) -> Optional[Dict]:
        """Detect multiple claims within short time window."""
        if len(transactions) < 2:
            return None
            
        # Sort by date
        sorted_txns = sorted(transactions, key=lambda x: x['transaction_date'])
        
        # Check 1-hour windows
        for i in range(len(sorted_txns) - 1):
            current = datetime.fromisoformat(sorted_txns[i]['transaction_date'].replace('Z', '+00:00'))
            next_txn = datetime.fromisoformat(sorted_txns[i+1]['transaction_date'].replace('Z', '+00:00'))
            
            time_diff = (next_txn - current).total_seconds() / 3600  # hours
            
            if time_diff <= 1:  # Within 1 hour
                # Count how many in this 1-hour window
                window_count = 2
                for j in range(i+2, len(sorted_txns)):
                    later = datetime.fromisoformat(sorted_txns[j]['transaction_date'].replace('Z', '+00:00'))
                    if (later - current).total_seconds() / 3600 <= 1:
                        window_count += 1
                    else:
                        break
                
                if window_count >= self.hourly_velocity_limit:
                    return {
                        'type': 'velocity_attack',
                        'reason': f"{window_count} claims within 1 hour (Velocity Attack)",
                        'score_contribution': min(95, 70 + (window_count - 2) * 8),
                        'count': window_count,
                        'time_window': '1 hour',
                        'feature': 'claim_velocity',
                        'evidence': f"Rapid successive claims detected"
                    }
        
        # Check daily velocity
        daily_counts = defaultdict(int)
        for txn in transactions:
            day = txn['transaction_date'][:10]  # YYYY-MM-DD
            daily_counts[day] += 1
        
        max_daily = max(daily_counts.values(), default=0)
        if max_daily > self.daily_claim_limit:
            return {
                'type': 'daily_velocity_exceeded',
                'reason': f"{max_daily} claims in single day (Limit: {self.daily_claim_limit})",
                'score_contribution': min(85, 60 + (max_daily - self.daily_claim_limit) * 12),
                'count': max_daily,
                'feature': 'daily_frequency',
                'evidence': f"Peak day had {max_daily} transactions"
            }
        
        return None
    
    def _check_off_hours(self, transactions: List[Dict]) -> Optional[Dict]:
        """Detect transactions during suspicious hours (10PM-5AM)."""
        off_hour_count = 0
        suspicious_transactions = []
        
        for txn in transactions:
            hour = int(txn['transaction_date'][11:13])  # Extract HH from timestamp
            
            if hour >= self.off_hours_start or hour < self.off_hours_end:
                off_hour_count += 1
                suspicious_transactions.append({
                    'time': txn['transaction_date'],
                    'hour': hour,
                    'amount': txn['amount']
                })
        
        if off_hour_count > 0:
            ratio = off_hour_count / len(transactions)
            
            # Score based on ratio and count
            if off_hour_count >= 3 and ratio > 0.3:
                score = 85
            elif off_hour_count >= 2:
                score = 70
            else:
                score = 55
            
            return {
                'type': 'off_hours_transaction',
                'reason': f"{off_hour_count} transactions during off-hours (10PM-5AM)",
                'score_contribution': score,
                'count': off_hour_count,
                'ratio': round(ratio, 2),
                'feature': 'temporal_pattern',
                'evidence': f"Transactions at odd hours: {[t['hour'] for t in suspicious_transactions[:3]]}"
            }
        return None
    
    def _check_frequency_anomaly(self, beneficiary_id: str, transactions: List[Dict]) -> Optional[Dict]:
        """Check if claim frequency deviates significantly from baseline."""
        if len(transactions) < 2:
            return None
            
        # Monthly rate
        days_span = 90
        monthly_rate = (len(transactions) / days_span) * 30
        
        baseline = self.baselines.get('avg_monthly_claims', 4)
        std_dev = self.baselines.get('std_monthly_claims', 2)
        
        # Z-score calculation
        if std_dev > 0:
            z_score = (monthly_rate - baseline) / std_dev
        else:
            z_score = 0
        
        if z_score > 2.5:  # >2.5 standard deviations above mean
            excess = monthly_rate - baseline
            return {
                'type': 'frequency_anomaly',
                'reason': f"Claim rate {monthly_rate:.1f}/month vs avg {baseline:.1f} (Z-score: {z_score:.1f})",
                'score_contribution': min(90, 60 + int(z_score * 10)),
                'monthly_rate': round(monthly_rate, 1),
                'baseline': baseline,
                'z_score': round(z_score, 2),
                'feature': 'frequency_deviation',
                'evidence': f"Excessive activity: {len(transactions)} claims in 90 days"
            }
        return None
    
    def _check_geographic_impossibility(self, transactions: List[Dict]) -> Optional[Dict]:
        """Check for impossible travel speeds between transactions."""
        if len(transactions) < 2:
            return None
            
        # Filter transactions with coordinates
        geo_txns = [t for t in transactions if t.get('latitude') and t.get('longitude')]
        if len(geo_txns) < 2:
            return None
            
        # Sort by time
        sorted_geo = sorted(geo_txns, key=lambda x: x['transaction_date'])
        
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km."""
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c
        
        violations = []
        
        for i in range(len(sorted_geo) - 1):
            t1, t2 = sorted_geo[i], sorted_geo[i+1]
            
            time1 = datetime.fromisoformat(t1['transaction_date'].replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(t2['transaction_date'].replace('Z', '+00:00'))
            time_diff_hours = (time2 - time1).total_seconds() / 3600
            
            if time_diff_hours < 0.5:  # Less than 30 minutes
                distance = haversine_distance(
                    t1['latitude'], t1['longitude'],
                    t2['latitude'], t2['longitude']
                )
                
                if time_diff_hours > 0:  # Avoid division by zero
                    speed = distance / time_diff_hours
                    
                    if speed > self.max_travel_speed and distance > 50:  # >100km/h and >50km apart
                        violations.append({
                            'from': t1['transaction_date'],
                            'to': t2['transaction_date'],
                            'distance_km': round(distance, 1),
                            'speed_kmh': round(speed, 1),
                            'time_diff_h': round(time_diff_hours, 2)
                        })
        
        if violations:
            worst = max(violations, key=lambda x: x['speed_kmh'])
            return {
                'type': 'geographic_impossibility',
                'reason': f"Impossible travel: {worst['distance_km']}km in {worst['time_diff_h']}h ({worst['speed_kmh']} km/h)",
                'score_contribution': min(100, 80 + len(violations) * 5),
                'violation_count': len(violations),
                'worst_violation': worst,
                'feature': 'geo_velocity',
                'evidence': f"Required {worst['speed_kmh']} km/h travel speed"
            }
        return None
    
    def _check_round_trip_fraud(self, transactions: List[Dict]) -> Optional[Dict]:
        """Detect rapid claim at location A, then B, then A (proxy/fake presence)."""
        if len(transactions) < 3:
            return None
            
        sorted_txns = sorted(transactions, key=lambda x: x['transaction_date'])
        
        # Look for A -> B -> A pattern within 24 hours
        for i in range(len(sorted_txns) - 2):
            t1 = sorted_txns[i]
            t2 = sorted_txns[i+1]
            t3 = sorted_txns[i+2]
            
            # Check if t1 and t3 are same agent (or nearby), t2 is different
            if t1['agent_id'] == t3['agent_id'] and t1['agent_id'] != t2['agent_id']:
                time1 = datetime.fromisoformat(t1['transaction_date'].replace('Z', '+00:00'))
                time3 = datetime.fromisoformat(t3['transaction_date'].replace('Z', '+00:00'))
                round_trip_time = (time3 - time1).total_seconds() / 3600  # hours
                
                if round_trip_time <= 24:  # Round trip within 24 hours
                    return {
                        'type': 'round_trip_fraud',
                        'reason': f"Round-trip fraud: Location {t1['agent_id']} -> {t2['agent_id']} -> {t1['agent_id']} in {round_trip_time:.1f}h",
                        'score_contribution': 75,
                        'time_hours': round(round_trip_time, 1),
                        'locations': [t1['agent_id'], t2['agent_id'], t3['agent_id']],
                        'feature': 'round_trip_pattern',
                        'evidence': "Suspicious round-trip claim pattern"
                    }
        return None
    
    def _check_office_hours_violation(self, transactions: List[Dict]) -> Optional[Dict]:
        """Check for claims during government holidays or Sundays."""
        weekend_count = 0
        
        for txn in transactions:
            date_obj = datetime.fromisoformat(txn['transaction_date'].replace('Z', '+00:00'))
            if date_obj.weekday() >= 5:  # Saturday=5, Sunday=6
                weekend_count += 1
        
        if weekend_count >= 2:  # Multiple weekend claims
            ratio = weekend_count / len(transactions)
            return {
                'type': 'weekend_transaction',
                'reason': f"{weekend_count} claims on weekends/holidays",
                'score_contribution': min(70, 50 + weekend_count * 5),
                'count': weekend_count,
                'ratio': round(ratio, 2),
                'feature': 'office_hours',
                'evidence': "Government offices closed during these claims"
            }
        return None
    
    def _score_to_severity(self, score: float) -> str:
        """Convert score to severity."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        return "low"

# Test
if __name__ == "__main__":
    engine = VelocityEngine()
    print(f"✓ Velocity Engine initialized")
    print(f"Baselines: {engine.baselines}")
    
    # Test analysis
    with engine.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT beneficiary_id FROM beneficiaries WHERE (SELECT COUNT(*) FROM transactions t WHERE t.beneficiary_id = beneficiaries.beneficiary_id) > 3 LIMIT 1")
        row = cursor.fetchone()
        if row:
            result = engine.analyze(row[0])
            print(f"\nTest Result: {result.score} - {result.description}")
            if result.details.get('violations'):
                for v in result.details['violations']:
                    print(f"  - {v['type']}: {v['reason'][:60]}...")