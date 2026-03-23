from core.pipeline import FraudDetectionPipeline

print("🔍 Testing fraud detection system...")

pipeline = FraudDetectionPipeline()

# Test 5 random beneficiaries
import random
test_ids = [f"BEN{random.randint(0, 99994):08d}" for _ in range(5)]

print(f"\nAnalyzing {len(test_ids)} cases:\n")

for bid in test_ids:
    report = pipeline.analyze(bid)
    print(f"{bid}: Score={report.overall_score:.1f} | Risk={report.risk_level} | "
          f"Rule={report.rule_score:.0f}, ML={report.ml_score:.0f}")

high = sum(1 for r in [pipeline.analyze(f"BEN{i:08d}") for i in random.sample(range(100000), 20)] if r.risk_level == 'High')
print(f"\n✅ System working! Found {high}/20 high-risk cases in sample.")
print("🚀 Start UI: streamlit run app.py")