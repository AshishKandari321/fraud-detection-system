# fix_database.py - Run once to add missing column
import sqlite3

conn = sqlite3.connect("data/processed/fraud_system.db")
cursor = conn.cursor()

# Add missing column
cursor.execute("ALTER TABLE fraud_results ADD COLUMN recommended_action TEXT")

conn.commit()
conn.close()
print("✓ Database fixed")