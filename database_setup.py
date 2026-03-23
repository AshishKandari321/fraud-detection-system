import os
import sqlite3

def setup_database():
    db_dir = "data/processed"
    db_path = os.path.join(db_dir, "fraud_system.db")
    os.makedirs(db_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS beneficiaries (
                beneficiary_id TEXT PRIMARY KEY, aadhaar_hash TEXT, aadhaar_masked TEXT,
                name TEXT, address TEXT, phone_hash TEXT, phone_masked TEXT,
                bank_hash TEXT, bank_masked TEXT, ifsc_code TEXT, annual_income REAL,
                scheme_type TEXT, district TEXT, state TEXT, pincode TEXT,
                family_size INTEGER, registration_date TEXT, status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY, beneficiary_id TEXT, scheme_type TEXT,
                amount REAL, timestamp TEXT, status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY, agent_type TEXT, fraud_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fraud_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT, beneficiary_id TEXT UNIQUE,
                overall_score REAL, risk_level TEXT, rule_score REAL DEFAULT 0,
                velocity_score REAL DEFAULT 0, ml_score REAL DEFAULT 0,
                anomaly_score REAL DEFAULT 0, graph_score REAL DEFAULT 0,
                primary_reasons TEXT, recommended_action TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

if __name__ == "__main__":
    setup_database()
