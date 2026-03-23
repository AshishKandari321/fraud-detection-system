# core/pipeline.py
"""Main fraud detection pipeline - orchestrates all engines and computes hybrid scores."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from core.models.beneficiary import FraudReport, RiskLevel, FraudIndicator
from core.engines.base import BaseDetectionEngine
from core.engines.rule_engine import RuleBasedEngine
from core.engines.velocity_engine import VelocityEngine
from core.engines.graph_engine import GraphEngine


class FraudDetectionPipeline:
    """
    Hybrid fraud detection pipeline.
    Combines: Rule-based (30%) + Velocity (25%) + Graph (20%) + ML (15%) + Anomaly (10%)
    """
    
    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        'rule': 0.30,
        'velocity': 0.25,
        'graph': 0.20,
        'ml': 0.15,
        'anomaly': 0.10
    }
    
    # Risk thresholds
    HIGH_RISK_THRESHOLD = 55    # Was 70
    MEDIUM_RISK_THRESHOLD = 30  # Was 40
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", 
                 weights: Optional[Dict[str, float]] = None):
        self.db_path = db_path
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.engines: Dict[str, BaseDetectionEngine] = {}
        self._initialize_engines()
        
    def _initialize_engines(self):
        """Initialize all detection engines with proper weights."""
        
        # Rule-based engine (deterministic)
        if self.weights.get('rule', 0) > 0:
            self.engines['rule'] = RuleBasedEngine(
                db_path=self.db_path, 
                weight=self.weights['rule']
            )
            print(f"✓ Rule Engine initialized (weight: {self.weights['rule']})")
        
        # Velocity engine (temporal)
        if self.weights.get('velocity', 0) > 0:
            self.engines['velocity'] = VelocityEngine(
                db_path=self.db_path,
                weight=self.weights['velocity']
            )
            self.engines['velocity'].train()
            print(f"✓ Velocity Engine initialized (weight: {self.weights['velocity']})")
        
        # Graph engine (network)
        if self.weights.get('graph', 0) > 0:
            self.engines['graph'] = GraphEngine(
                db_path=self.db_path,
                weight=self.weights['graph']
            )
            self.engines['graph'].train()
            print(f"✓ Graph Engine initialized (weight: {self.weights['graph']})")
        
        # ML Engine (supervised learning)
        if self.weights.get('ml', 0) > 0:
            from core.engines.ml_engine import MLEngine
            self.engines['ml'] = MLEngine(
                db_path=self.db_path,
                weight=self.weights['ml']
            )
            self.engines['ml'].train()
            print(f"✓ ML Engine initialized (weight: {self.weights['ml']})")
        
        # Anomaly Engine (unsupervised)
        if self.weights.get('anomaly', 0) > 0:
            from core.engines.anomaly_engine import AnomalyEngine
            self.engines['anomaly'] = AnomalyEngine(
                db_path=self.db_path,
                weight=self.weights['anomaly']
            )
            self.engines['anomaly'].train()
            print(f"✓ Anomaly Engine initialized (weight: {self.weights['anomaly']})")
        
        print(f"✓ Pipeline ready with {len(self.engines)} engines")
    
    def analyze(self, beneficiary_id: str) -> FraudReport:
        """
        Run complete fraud analysis using all registered engines.
        
        Args:
            beneficiary_id: Unique beneficiary ID
            
        Returns:
            FraudReport with composite score and explanations
        """
        # Verify beneficiary exists
        if not self._validate_beneficiary(beneficiary_id):
            return FraudReport(
                beneficiary_id=beneficiary_id,
                overall_score=0,
                risk_level=RiskLevel.LOW,
                primary_reasons=["Beneficiary not found in database"],
                recommended_action="Verify beneficiary ID and try again"
            )
        
        # Collect indicators from all engines
        indicators = []
        engine_scores = {}
        
        print(f"\n🔍 Analyzing {beneficiary_id}...")
        
        # Sequential execution (for debugging) or parallel (for production)
        for name, engine in self.engines.items():
            try:
                print(f"  Running {name} engine...")
                indicator = engine.analyze(beneficiary_id)
                indicators.append(indicator)
                engine_scores[name] = indicator.score
                print(f"    Score: {indicator.score:.1f} ({indicator.severity})")
            except Exception as e:
                print(f"    ⚠️  Error in {name} engine: {e}")
                # Add neutral indicator on error
                indicators.append(FraudIndicator(
                    engine=name,
                    score=50,
                    severity="medium",
                    description=f"{name} engine error: {str(e)}"
                ))
                engine_scores[name] = 50
        
        # Calculate weighted composite score
        overall_score = self._calculate_hybrid_score(indicators)
        
        # Determine risk level
        risk_level = self._calculate_risk_level(overall_score)
        
        # Extract primary reasons (top 3 highest scoring violations)
        primary_reasons = self._extract_primary_reasons(indicators)
        
        # Generate recommendations
        recommended_action = self._generate_recommendation(risk_level, primary_reasons)
        
        # Create report
        report = FraudReport(
            beneficiary_id=beneficiary_id,
            timestamp=datetime.now(),
            overall_score=round(overall_score, 2),
            risk_level=risk_level,
            rule_score=engine_scores.get('rule', 0),
            velocity_score=engine_scores.get('velocity', 0),
            ml_score=engine_scores.get('ml', 0),
            anomaly_score=engine_scores.get('anomaly', 0),
            graph_score=engine_scores.get('graph', 0),
            indicators=indicators,
            primary_reasons=primary_reasons,
            recommended_action=recommended_action
        )
        
        # Save to database
        self._save_result(report)
        
        return report
    
    def analyze_batch(self, beneficiary_ids: List[str], 
                     max_workers: int = 4) -> List[FraudReport]:
        """
        Batch analysis with parallel processing.
        
        Args:
            beneficiary_ids: List of IDs to analyze
            max_workers: Parallel threads
            
        Returns:
            List of FraudReports
        """
        results = []
        
        print(f"\n🚀 Batch analysis of {len(beneficiary_ids)} beneficiaries...")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.analyze, bid): bid 
                for bid in beneficiary_ids
            }
            
            for future in as_completed(future_to_id):
                bid = future_to_id[future]
                try:
                    report = future.result()
                    results.append(report)
                except Exception as e:
                    print(f"Error analyzing {bid}: {e}")
                    # Add error report
                    results.append(FraudReport(
                        beneficiary_id=bid,
                        overall_score=0,
                        risk_level=RiskLevel.LOW,
                        primary_reasons=[f"Analysis error: {str(e)}"],
                        recommended_action="Retry analysis"
                    ))
        
        # Sort by beneficiary ID for consistency
        results.sort(key=lambda x: x.beneficiary_id)
        
        print(f"\n✓ Batch complete: {len(results)} analyzed")
        
        # Print summary
        high_risk = sum(1 for r in results if r.risk_level == RiskLevel.HIGH)
        medium_risk = sum(1 for r in results if r.risk_level == RiskLevel.MEDIUM)
        print(f"  High Risk: {high_risk} | Medium: {medium_risk} | Low: {len(results) - high_risk - medium_risk}")
        
        return results
    
    def _calculate_hybrid_score(self, indicators):
        """Final corrected scoring - Rule 30%, Velocity 25%, Graph 20%, ML 15%, Anomaly 10%"""
        scores = {}
        for ind in indicators:
            scores[ind.engine] = ind.score
    
        # Handle name variations and defaults
        s_rule = scores.get('rule', scores.get('rule_based', 0))
        s_vel = scores.get('velocity', 0)
        s_graph = scores.get('graph', 0)
        s_ml = scores.get('ml', 0)
        s_anom = scores.get('anomaly', 0)
    
        # Weighted average (0-100 scale)
        total = (s_rule * 0.30) + (s_vel * 0.25) + (s_graph * 0.20) + (s_ml * 0.15) + (s_anom * 0.10)
    
        return float(total)
    def _calculate_risk_level(self, score: float) -> RiskLevel:
        """Categorize risk based on score thresholds."""
        if score >= self.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif score >= self.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _extract_primary_reasons(self, indicators: List[FraudIndicator], max_reasons: int = 3) -> List[str]:
        """Extract top reasons for fraud flag."""
        all_violations = []
        
        for ind in indicators:
            if hasattr(ind, 'details') and ind.details and 'violations' in ind.details:
                for v in ind.details['violations']:
                    all_violations.append({
                        'reason': v.get('reason', ind.description),
                        'score': v.get('score_contribution', ind.score),
                        'engine': ind.engine
                    })
        
        # Sort by score contribution
        all_violations.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top N unique reasons
        seen = set()
        reasons = []
        for v in all_violations:
            if v['reason'] not in seen and len(reasons) < max_reasons:
                reasons.append(f"[{v['engine']}] {v['reason']}")
                seen.add(v['reason'])
        
        return reasons if reasons else ["No significant violations detected"]
    
    def _generate_recommendation(self, risk_level: RiskLevel, reasons: List[str]) -> str:
        """Generate actionable recommendation."""
        recommendations = {
            RiskLevel.HIGH: "IMMEDIATE ACTION REQUIRED: Suspend beneficiary and initiate detailed investigation. High probability of fraud detected.",
            RiskLevel.MEDIUM: "ENHANCED MONITORING: Verify documents and monitor transactions closely. Schedule follow-up review within 7 days.",
            RiskLevel.LOW: "STANDARD PROCESSING: Continue with routine verification. No immediate action required."
        }
        
        base_rec = recommendations.get(risk_level, "Review case manually")
        
        # Add specific guidance based on reasons
        if any('duplicate' in r.lower() for r in reasons):
            base_rec += " | Verify identity documents against Aadhaar database."
        if any('velocity' in r.lower() or 'off-hours' in r.lower() for r in reasons):
            base_rec += " | Review transaction timing patterns for automation/bots."
        if any('graph' in r.lower() or 'network' in r.lower() for r in reasons):
            base_rec += " | Map connections to identify broader fraud ring."
        
        return base_rec
    
    def _validate_beneficiary(self, beneficiary_id: str) -> bool:
        """Check if beneficiary exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM beneficiaries WHERE beneficiary_id = ?",
                    (beneficiary_id,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def _save_result(self, report: FraudReport):
        """Save analysis result to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO fraud_results 
                    (beneficiary_id, overall_score, risk_level, rule_score, velocity_score,
                     ml_score, anomaly_score, graph_score, primary_reasons, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.beneficiary_id,
                    report.overall_score,
                    report.risk_level,
                    report.rule_score,
                    report.velocity_score,
                    report.ml_score,
                    report.anomaly_score,
                    report.graph_score,
                    json.dumps(report.primary_reasons),
                    report.recommended_action
                ))
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not save result: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT risk_level, COUNT(*) 
                    FROM fraud_results 
                    GROUP BY risk_level
                """)
                risk_distribution = {row[0]: row[1] for row in cursor.fetchall()}
                
                cursor.execute("SELECT COUNT(*) FROM fraud_results")
                total_analyzed = cursor.fetchone()[0]
                
                return {
                    'total_analyzed': total_analyzed,
                    'risk_distribution': risk_distribution,
                    'engines_active': list(self.engines.keys()),
                    'weights': self.weights
                }
        except Exception as e:
            return {'error': str(e)}


# Test
if __name__ == "__main__":
    print("="*60)
    print("FRAUD DETECTION PIPELINE TEST")
    print("="*60)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Test single analysis
    with sqlite3.connect(pipeline.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT beneficiary_id FROM beneficiaries LIMIT 5")
        test_ids = [r[0] for r in cursor.fetchall()]
    
    print(f"\nTesting with {len(test_ids)} beneficiaries:\n")
    
    for bid in test_ids:
        report = pipeline.analyze(bid)
        print(f"\n📊 Result for {bid}:")
        print(f"   Overall Score: {report.overall_score}/100")
        print(f"   Risk Level: {report.risk_level}")
        print(f"   Breakdown: Rule={report.rule_score:.0f}, Velocity={report.velocity_score:.0f}, ML={report.ml_score:.0f}, Anomaly={report.anomaly_score:.0f}, Graph={report.graph_score:.0f}")
        print(f"   Primary Reasons:")
        for reason in report.primary_reasons[:2]:
            print(f"      • {reason}")
        print(f"   Action: {report.recommended_action[:80]}...")
    
    # Statistics
    print(f"\n{'='*60}")
    print("Pipeline Statistics:")
    stats = pipeline.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")