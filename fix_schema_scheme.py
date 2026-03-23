import sqlite3

conn = sqlite3.connect("data/processed/fraud_system.db")
cursor = conn.cursor()

# Check if column exists
cursor.execute("PRAGMA table_info(beneficiaries)")
columns = [col[1] for col in cursor.fetchall()]

if 'scheme_type' not in columns:
    print("Adding scheme_type column to beneficiaries...")
    cursor.execute("ALTER TABLE beneficiaries ADD COLUMN scheme_type TEXT")
    conn.commit()
    print("✅ Column added")
    
    # Update existing records with default scheme based on their transactions
    cursor.execute("""
        UPDATE beneficiaries 
        SET scheme_type = (
            SELECT scheme_type FROM transactions 
            WHERE transactions.beneficiary_id = beneficiaries.beneficiary_id 
            LIMIT 1
        )
        WHERE scheme_type IS NULL
    """)
    conn.commit()
    print("✅ Existing records updated")
else:
    print("✅ scheme_type already exists")

conn.close()
print("Done!")