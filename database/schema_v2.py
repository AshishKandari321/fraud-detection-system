# database/schema_v2.py
"""Production relational schema: 100K beneficiaries + 400K transactions + 5K agents."""
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager

class RelationalFraudDB:
    """
    SQLite database with foreign key relationships.
    Optimized for fraud detection queries on 100K+ records.
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db"):
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Test connection
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.close()
        
        self._init_tables()
        print(f"✓ Database initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_tables(self):
        """Create optimized schema with indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. BENEFICIARIES (100K records) - Master data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS beneficiaries (
                    beneficiary_id TEXT PRIMARY KEY,
                    aadhaar_hash TEXT NOT NULL,
                    aadhaar_masked TEXT NOT NULL,
                    name TEXT NOT NULL,
                    address TEXT NOT NULL,
                    phone_hash TEXT,
                    phone_masked TEXT,
                    bank_hash TEXT NOT NULL,
                    bank_masked TEXT NOT NULL,
                    ifsc_code TEXT,
                    annual_income REAL CHECK (annual_income >= 0),
                    occupation TEXT,
                    family_size INTEGER,
                    district TEXT NOT NULL,
                    state TEXT NOT NULL,
                    pincode TEXT,
                    registration_date DATE NOT NULL,
                    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deceased', 'blocked')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. AGENTS/MERCHANTS (5K records) - PDS shops, gas agencies, etc.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL CHECK (agent_type IN ('pds_shop', 'gas_agency', 'bank_branch', 'online_portal', 'CSC')),
                    name TEXT NOT NULL,
                    district TEXT NOT NULL,
                    state TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    license_number TEXT,
                    fraud_score REAL DEFAULT 0 CHECK (fraud_score >= 0 AND fraud_score <= 100),
                    status TEXT DEFAULT 'active',
                    registration_date DATE DEFAULT CURRENT_DATE
                )
            """)
            
            # 3. TRANSACTIONS (400K records) - Claims/disbursements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    beneficiary_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    scheme_type TEXT NOT NULL CHECK (scheme_type IN ('PDS', 'PAHAL', 'PM_KISAN', 'PENSION', 'SCHOLARSHIP')),
                    amount REAL NOT NULL CHECK (amount > 0),
                    transaction_date TIMESTAMP NOT NULL,
                    channel TEXT CHECK (channel IN ('online', 'offline', 'AEPS', 'UPI', 'NEFT', 'cash')),
                    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'reversed', 'pending')),
                    latitude REAL,
                    longitude REAL,
                    device_id TEXT,
                    ip_address TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (beneficiary_id) REFERENCES beneficiaries(beneficiary_id) ON DELETE CASCADE,
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
                )
            """)
            
            # 4. EVENTS (Audit trail) - Login attempts, changes, flags
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    beneficiary_id TEXT,
                    agent_id TEXT,
                    event_type TEXT NOT NULL CHECK (event_type IN ('login', 'claim_initiated', 'document_updated', 'verification_failed', 'flag_raised', 'investigation_opened', 'status_changed')),
                    severity TEXT DEFAULT 'info' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    ip_address TEXT,
                    FOREIGN KEY (beneficiary_id) REFERENCES beneficiaries(beneficiary_id),
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
                )
            """)
            
            # 5. FRAUD RESULTS (Cache for detection outputs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    beneficiary_id TEXT UNIQUE NOT NULL,
                    overall_score REAL CHECK (overall_score >= 0 AND overall_score <= 100),
                    risk_level TEXT CHECK (risk_level IN ('High', 'Medium', 'Low')),
                    rule_score REAL,
                    velocity_score REAL,
                    ml_score REAL,
                    anomaly_score REAL,
                    graph_score REAL,
                    primary_reasons TEXT,  -- JSON array
                    detected_patterns TEXT,  -- JSON array
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (beneficiary_id) REFERENCES beneficiaries(beneficiary_id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for performance (critical for 400K transactions)
            indices = [
                # Beneficiary lookups
                "CREATE INDEX IF NOT EXISTS idx_ben_aadhaar ON beneficiaries(aadhaar_hash)",
                "CREATE INDEX IF NOT EXISTS idx_ben_bank ON beneficiaries(bank_hash)",
                "CREATE INDEX IF NOT EXISTS idx_ben_phone ON beneficiaries(phone_hash)",
                "CREATE INDEX IF NOT EXISTS idx_ben_district ON beneficiaries(district)",
                "CREATE INDEX IF NOT EXISTS idx_ben_state ON beneficiaries(state)",
                "CREATE INDEX IF NOT EXISTS idx_ben_income ON beneficiaries(annual_income)",
                "CREATE INDEX IF NOT EXISTS idx_ben_status ON beneficiaries(status)",
                
                # Transaction analysis (velocity detection)
                "CREATE INDEX IF NOT EXISTS idx_txn_beneficiary ON transactions(beneficiary_id)",
                "CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date)",
                "CREATE INDEX IF NOT EXISTS idx_txn_agent ON transactions(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_txn_scheme ON transactions(scheme_type)",
                "CREATE INDEX IF NOT EXISTS idx_txn_amount ON transactions(amount)",
                "CREATE INDEX IF NOT EXISTS idx_txn_status ON transactions(status)",
                "CREATE INDEX IF NOT EXISTS idx_txn_geo ON transactions(latitude, longitude)",
                
                # Agent networks
                "CREATE INDEX IF NOT EXISTS idx_agent_type ON agents(agent_type)",
                "CREATE INDEX IF NOT EXISTS idx_agent_district ON agents(district)",
                "CREATE INDEX IF NOT EXISTS idx_agent_score ON agents(fraud_score)",
                
                # Events audit
                "CREATE INDEX IF NOT EXISTS idx_events_ben ON events(beneficiary_id)",
                "CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
                
                # Composite index for time-series fraud queries
                "CREATE INDEX IF NOT EXISTS idx_txn_ben_date ON transactions(beneficiary_id, transaction_date)",
            ]
            
            for idx_sql in indices:
                cursor.execute(idx_sql)
            
            conn.commit()
    
    def bulk_insert_beneficiaries(self, records: List[Dict], batch_size: int = 5000):
        """Insert beneficiaries with privacy masking."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = """
                INSERT OR REPLACE INTO beneficiaries 
                (beneficiary_id, aadhaar_hash, aadhaar_masked, name, address,
                 phone_hash, phone_masked, bank_hash, bank_masked, ifsc_code,
                 annual_income, occupation, family_size, district, state, pincode, 
                 registration_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                values = [(
                    r['beneficiary_id'], r['aadhaar_hash'], r['aadhaar_masked'],
                    r['name'], r['address'], r['phone_hash'], r['phone_masked'],
                    r['bank_hash'], r['bank_masked'], r['ifsc_code'],
                    r['annual_income'], r.get('occupation'), r.get('family_size', 4),
                    r['district'], r['state'], r['pincode'],
                    r['registration_date'], r.get('status', 'active')
                ) for r in batch]
                
                cursor.executemany(sql, values)
                conn.commit()
                print(f"  Inserted {min(i+batch_size, len(records)):,}/{len(records):,} beneficiaries")
    
    def bulk_insert_agents(self, records: List[Dict], batch_size: int = 1000):
        """Insert agents/merchants."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = """
                INSERT OR REPLACE INTO agents 
                (agent_id, agent_type, name, district, state, latitude, longitude, 
                 license_number, fraud_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                values = [(
                    r['agent_id'], r['agent_type'], r['name'], r['district'],
                    r['state'], r.get('latitude'), r.get('longitude'),
                    r.get('license_number'), r.get('fraud_score', 0), r.get('status', 'active')
                ) for r in batch]
                
                cursor.executemany(sql, values)
                conn.commit()
    
    def bulk_insert_transactions(self, records: List[Dict], batch_size: int = 10000):
        """Insert transactions (largest table - optimized)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = """
                INSERT OR REPLACE INTO transactions 
                (transaction_id, beneficiary_id, agent_id, scheme_type, amount,
                 transaction_date, channel, status, latitude, longitude, device_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                values = [(
                    r['transaction_id'], r['beneficiary_id'], r['agent_id'],
                    r['scheme_type'], r['amount'], r['transaction_date'],
                    r.get('channel', 'offline'), r.get('status', 'success'),
                    r.get('latitude'), r.get('longitude'), r.get('device_id')
                ) for r in batch]
                
                cursor.executemany(sql, values)
                conn.commit()
                if (i // batch_size) % 5 == 0:  # Log every 5th batch
                    print(f"  Inserted {min(i+batch_size, len(records)):,}/{len(records):,} transactions")
    
    def get_statistics(self) -> Dict:
        """Get quick stats about the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            for table in ['beneficiaries', 'agents', 'transactions', 'events']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            # Fraud-specific stats
            cursor.execute("SELECT COUNT(DISTINCT aadhaar_hash) FROM beneficiaries")
            stats['unique_aadhaars'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT scheme_type, COUNT(*), SUM(amount) 
                FROM transactions 
                WHERE status='success' 
                GROUP BY scheme_type
            """)
            stats['scheme_wise'] = cursor.fetchall()
            
            return stats

# Test/Initialize
if __name__ == "__main__":
    db = RelationalFraudDB()
    stats = db.get_statistics()
    print(f"\n📊 Current Database Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")