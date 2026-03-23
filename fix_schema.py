import sqlite3
import os

db_path = "data/processed/fraud_system.db"

if not os.path.exists(db_path):
    print("❌ Database not found! Run: python data_generator/relational_generator.py")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if fraud_results exists and has all columns
cursor.execute("PRAGMA table_info(fraud_results)")
columns = [col[1] for col in cursor.fetchall()]

required = ['beneficiary_id', 'overall_score', 'risk_level', 'rule_score', 
           'velocity_score', 'ml_score', 'anomaly_score', 'graph_score', 
           'primary_reasons', 'recommended_action', 'analyzed_at']

missing = [r for r in required if r not in columns]

if missing:
    print(f"⚠️  Missing columns: {missing}")
    print("🗑️  Dropping and recreating fraud_results table...")
    cursor.execute("DROP TABLE IF EXISTS fraud_results")
    cursor.execute("""
        CREATE TABLE fraud_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            beneficiary_id TEXT UNIQUE NOT NULL,
            overall_score REAL,
            risk_level TEXT,
            rule_score REAL DEFAULT 0,
            velocity_score REAL DEFAULT 0,
            ml_score REAL DEFAULT 0,
            anomaly_score REAL DEFAULT 0,
            graph_score REAL DEFAULT 0,
            primary_reasons TEXT,
            recommended_action TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("✅ Schema fixed!")
else:
    print("✅ Schema looks good")

conn.close()
print("\n🚀 Ready to run: streamlit run app.py")