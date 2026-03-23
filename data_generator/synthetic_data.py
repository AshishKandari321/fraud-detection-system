
"""Generate 200,000 realistic Indian beneficiary records with embedded fraud patterns."""
import random
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from faker import Faker
import polars as pl
import numpy as np
from tqdm import tqdm
import os

fake = Faker('en_IN')

class DemographicConfig:
    """Real-world Indian demographic distributions based on census data."""
    
    # Major states with population weights
    STATES_DISTRICTS = {
        'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Ghaziabad', 'Agra', 'Varanasi', 'Prayagraj', 'Meerut', 'Noida'],
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad', 'Solapur'],
        'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga', 'Arrah'],
        'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri', 'Malda'],
        'Madhya Pradesh': ['Indore', 'Bhopal', 'Jabalpur', 'Gwalior', 'Ujjain'],
        'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli'],
        'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer', 'Bikaner'],
        'Karnataka': ['Bengaluru', 'Mysuru', 'Hubli', 'Mangalore', 'Belgaum'],
        'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar'],
        'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore'],
        'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar'],
        'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur'],
        'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Puri'],
        'Jharkhand': ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro'],
        'Assam': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat'],
        'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala'],
        'Delhi': ['New Delhi', 'North Delhi', 'South Delhi', 'East Delhi']
    }
    
    # Income distribution by state (mean, std in INR)
    INCOME_PROFILES = {
        'Delhi': (450000, 200000), 'Maharashtra': (380000, 180000),
        'Karnataka': (360000, 170000), 'Telangana': (340000, 160000),
        'Gujarat': (320000, 150000), 'Tamil Nadu': (310000, 140000),
        'Kerala': (300000, 140000), 'Haryana': (290000, 130000),
        'Punjab': (285000, 130000), 'West Bengal': (220000, 100000),
        'Rajasthan': (210000, 95000), 'Andhra Pradesh': (230000, 105000),
        'Odisha': (190000, 85000), 'Assam': (180000, 80000),
        'Jharkhand': (175000, 78000), 'Bihar': (160000, 70000),
        'Uttar Pradesh': (170000, 75000), 'Madhya Pradesh': (185000, 82000)
    }
    
    # Scheme distribution based on real DBT allocation
    SCHEMES = {
        'PDS': 0.40,           # 40% - Food subsidy (largest)
        'PAHAL': 0.25,         # 25% - LPG subsidy
        'PM_KISAN': 0.20,      # 20% - Farmer income support
        'PENSION': 0.10,       # 10% - Social security
        'SCHOLARSHIP': 0.05    # 5% - Education aid
    }
    
    # Eligibility criteria
    INCOME_THRESHOLDS = {
        'PDS': 300000,         # BPL approx
        'PAHAL': float('inf'), # Universal (but can check for multiple connections)
        'PM_KISAN': 2000000,   # Land owners with income check (relaxed)
        'PENSION': 200000,     # Low income seniors
        'SCHOLARSHIP': 800000  # Family income for non-merit
    }

class SyntheticDataGenerator:
    """Generate 200K beneficiaries with realistic fraud injection."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)
        
        self.states = list(DemographicConfig.STATES_DISTRICTS.keys())
        self.used_aadhaars = set()
        self.used_banks = set()
        self.fraud_clusters = []
        
    def _generate_aadhaar(self) -> str:
        """Generate unique 12-digit Aadhaar."""
        while True:
            aadhaar = ''.join([str(random.randint(0, 9)) for _ in range(12)])
            if aadhaar not in self.used_aadhaars:
                self.used_aadhaars.add(aadhaar)
                return aadhaar
    
    def _generate_bank_account(self) -> Tuple[str, str]:
        """Generate unique bank account with IFSC."""
        while True:
            # Major Indian banks
            bank_code = random.choice(['SBIN', 'HDFC', 'ICIC', 'PNB', 'BOB', 'CBI', 'UCO', 'AXIS'])
            branch = random.randint(10000, 99999)
            account = ''.join([str(random.randint(0, 9)) for _ in range(12)])
            ifsc = f"{bank_code}0{branch:05d}"
            
            if account not in self.used_banks:
                self.used_banks.add(account)
                return account, ifsc
    
    def _generate_income(self, state: str) -> float:
        """Generate income based on state distribution."""
        mean, std = DemographicConfig.INCOME_PROFILES.get(state, (250000, 100000))
        income = np.random.normal(mean, std)
        return max(50000, min(income, 5000000))  # Clamp between 50K and 50L
    
    def _select_scheme(self, income: float) -> Tuple[str, float]:
        """Select scheme based on probability and income eligibility."""
        scheme = random.choices(
            list(DemographicConfig.SCHEMES.keys()),
            weights=list(DemographicConfig.SCHEMES.values())
        )[0]
        
        # Subsidy amounts
        subsidy_map = {
            'PDS': random.uniform(1000, 3000),
            'PAHAL': random.uniform(800, 1200),
            'PM_KISAN': 6000,  # Fixed yearly
            'PENSION': random.uniform(1000, 2000),
            'SCHOLARSHIP': random.uniform(5000, 20000)
        }
        
        return scheme, subsidy_map[scheme]
    
    def generate_legitimate_record(self, index: int) -> Dict:
        """Generate one legitimate beneficiary."""
        state = random.choice(self.states)
        district = random.choice(DemographicConfig.STATES_DISTRICTS[state])
        
        income = self._generate_income(state)
        scheme, subsidy = self._select_scheme(income)
        
        # Registration date over last 5 years
        days_back = random.randint(1, 365*5)
        reg_date = datetime.now() - timedelta(days=days_back)
        
        aadhaar = self._generate_aadhaar()
        account, ifsc = self._generate_bank_account()
        
        return {
            'beneficiary_id': f"BEN{index:08d}",
            'aadhaar_number': aadhaar,
            'name': fake.name(),
            'address': fake.address().replace('\n', ', '),
            'phone_number': fake.msisdn()[2:12],  # 10 digits
            'bank_account': account,
            'ifsc_code': ifsc,
            'annual_income': round(income, 2),
            'scheme_type': scheme,
            'subsidy_amount': subsidy,
            'district': district,
            'state': state,
            'pincode': fake.postcode(),
            'registration_date': reg_date.strftime('%Y-%m-%d'),
            'is_active': 1,
            'is_fraud': 0,
            'fraud_type': None
        }
    
    def generate_fraud_cluster(self, base_index: int, cluster_type: str, size: int) -> List[Dict]:
        """
        Generate a coordinated fraud ring.
        Types: 'duplicate_aadhaar', 'shared_account', 'ghost_beneficiaries', 
               'income_mismatch', 'address_cluster'
        """
        records = []
        
        if cluster_type == 'duplicate_aadhaar':
            # Same person, multiple bank accounts/schemes
            master_aadhaar = self._generate_aadhaar()
            for i in range(size):
                rec = self.generate_legitimate_record(base_index + i)
                rec['aadhaar_number'] = master_aadhaar  # Override
                rec['is_fraud'] = 1
                rec['fraud_type'] = 'duplicate_aadhaar'
                records.append(rec)
                
        elif cluster_type == 'shared_account':
            # Multiple beneficiaries sharing one bank account (fraud ring)
            shared_account, shared_ifsc = self._generate_bank_account()
            shared_address = fake.address().replace('\n', ', ')
            
            for i in range(size):
                rec = self.generate_legitimate_record(base_index + i)
                rec['bank_account'] = shared_account
                rec['ifsc_code'] = shared_ifsc
                rec['address'] = shared_address
                rec['is_fraud'] = 1
                rec['fraud_type'] = 'shared_account_ring'
                records.append(rec)
                
        elif cluster_type == 'ghost_beneficiaries':
            # Fake identities, often with similar patterns
            common_surname = fake.last_name()
            for i in range(size):
                rec = self.generate_legitimate_record(base_index + i)
                rec['name'] = f"{fake.first_name()} {common_surname}"
                rec['address'] = f"Ghost Village {i}, {rec['district']}"  # Suspicious
                rec['annual_income'] = random.uniform(1000, 10000)  # Unrealistically low
                rec['is_fraud'] = 1
                rec['fraud_type'] = 'ghost_identity'
                records.append(rec)
                
        elif cluster_type == 'income_mismatch':
            # High income but claiming low-income subsidies
            for i in range(size):
                rec = self.generate_legitimate_record(base_index + i)
                rec['annual_income'] = random.uniform(1500000, 5000000)  # High income
                rec['scheme_type'] = random.choice(['PDS', 'PENSION'])  # Low-income schemes
                rec['is_fraud'] = 1
                rec['fraud_type'] = 'income_eligibility_fraud'
                records.append(rec)
                
        elif cluster_type == 'address_cluster':
            # Many beneficiaries at same address (above threshold)
            shared_address = fake.address().replace('\n', ', ')
            for i in range(size):
                rec = self.generate_legitimate_record(base_index + i)
                rec['address'] = shared_address
                rec['phone_number'] = random.choice([rec['phone_number'][:5] + "00000", rec['phone_number']])
                rec['is_fraud'] = 1
                rec['fraud_type'] = 'suspicious_address_cluster'
                records.append(rec)
        
        return records
    
    def generate_dataset(self, total_records: int = 200000, fraud_rate: float = 0.15) -> pl.DataFrame:
        """
        Generate full dataset with embedded fraud patterns.
        
        Fraud distribution:
        - 5% duplicate Aadhaar
        - 4% shared accounts (rings)
        - 2% ghost beneficiaries
        - 3% income mismatch
        - 1% address clusters (suspicious but possible)
        """
        print(f"Generating {total_records:,} beneficiary records...")
        
        all_records = []
        current_index = 0
        
        # Calculate fraud counts
        fraud_total = int(total_records * fraud_rate)
        legitimate_total = total_records - fraud_total
        
        # Fraud distribution
        fraud_config = [
            ('duplicate_aadhaar', int(fraud_total * 0.33)),      # 33% of fraud
            ('shared_account', int(fraud_total * 0.27)),         # 27% of fraud
            ('income_mismatch', int(fraud_total * 0.20)),        # 20% of fraud
            ('ghost_beneficiaries', int(fraud_total * 0.13)),    # 13% of fraud
            ('address_cluster', int(fraud_total * 0.07))         # 7% of fraud
        ]
        
        # Generate legitimate records in chunks
        print("Generating legitimate beneficiaries...")
        for i in tqdm(range(0, legitimate_total, 1000)):
            chunk = []
            for j in range(min(1000, legitimate_total - i)):
                chunk.append(self.generate_legitimate_record(current_index))
                current_index += 1
            all_records.extend(chunk)
        
        # Generate fraud clusters
        print("Injecting fraud patterns...")
        for fraud_type, count in fraud_config:
            clusters_needed = max(1, count // 5)  # Average 5 per cluster
            avg_size = count // clusters_needed
            
            for _ in tqdm(range(clusters_needed), desc=fraud_type):
                cluster_size = random.randint(max(2, avg_size-2), avg_size+2)
                cluster = self.generate_fraud_cluster(current_index, fraud_type, cluster_size)
                all_records.extend(cluster)
                current_index += len(cluster)
        
        # Shuffle to mix fraud and legitimate
        random.shuffle(all_records)
        
        # Convert to Polars for speed
        df = pl.DataFrame(all_records)
        
        print(f"\nDataset Generated:")
        print(f"Total Records: {len(df)}")
        print(f"Fraud Distribution:\n{df.group_by('fraud_type').agg(pl.count()).sort('count', descending=True)}")
        
        return df
    
    def save_to_database(self, df: pl.DataFrame, db_path: str = "data/processed/fraud_detection.db"):
        """Save generated data to SQLite with privacy masking."""
        import sqlite3
        from privacy.masker import DataMasker
        
        print("Saving to database with privacy masking...")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS beneficiaries (
                beneficiary_id TEXT PRIMARY KEY,
                aadhaar_hash TEXT,
                aadhaar_masked TEXT,
                name TEXT,
                address TEXT,
                phone_hash TEXT,
                phone_masked TEXT,
                bank_hash TEXT,
                bank_masked TEXT,
                ifsc_code TEXT,
                annual_income REAL,
                scheme_type TEXT,
                subsidy_amount REAL,
                district TEXT,
                state TEXT,
                pincode TEXT,
                registration_date TEXT,
                is_fraud INTEGER,
                fraud_type TEXT
            )
        """)
        
        # Create indices
        indices = ['aadhaar_hash', 'bank_hash', 'phone_hash', 'address', 'district', 'state']
        for idx in indices:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{idx} ON beneficiaries({idx})")
        
        # Insert in batches with masking
        batch_size = 5000
        records = df.to_dicts()
        
        for i in tqdm(range(0, len(records), batch_size)):
            batch = records[i:i+batch_size]
            values = []
            
            for rec in batch:
                # Hash sensitive data, mask for display
                values.append((
                    rec['beneficiary_id'],
                    DataMasker.hash_identifier(rec['aadhaar_number']),
                    DataMasker.mask_aadhaar(rec['aadhaar_number']),
                    rec['name'],
                    rec['address'],
                    DataMasker.hash_identifier(rec['phone_number']),
                    DataMasker.mask_phone(rec['phone_number']),
                    DataMasker.hash_identifier(rec['bank_account']),
                    DataMasker.mask_bank_account(rec['bank_account']),
                    rec['ifsc_code'],
                    rec['annual_income'],
                    rec['scheme_type'],
                    rec['subsidy_amount'],
                    rec['district'],
                    rec['state'],
                    rec['pincode'],
                    rec['registration_date'],
                    rec['is_fraud'],
                    rec['fraud_type']
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO beneficiaries 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
            conn.commit()
        
        conn.close()
        print(f"✓ Saved {len(records):,} records to {db_path}")
        
        # Save raw CSV for reference (optional)
        csv_path = "data/processed/beneficiaries_raw.csv"
        df.write_csv(csv_path)
        print(f"✓ Also saved CSV backup to {csv_path}")

# Usage script
if __name__ == "__main__":
    print("Initializing 200K Beneficiary Generator...")
    gen = SyntheticDataGenerator(seed=2024)
    
    # Generate 200,000 records with 15% fraud rate (~30K fraud cases)
    df = gen.generate_dataset(total_records=200000, fraud_rate=0.15)
    
    # Save to database
    gen.save_to_database(df)
    
    print("\n🎯 Data Generation Complete!")
    print("Next: Run detection engines...")