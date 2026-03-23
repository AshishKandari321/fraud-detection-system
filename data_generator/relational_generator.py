# data_generator/relational_generator.py
"""Generate 100K beneficiaries + 5K agents + 400K transactions with fraud patterns."""
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from faker import Faker
import numpy as np
from tqdm import tqdm
import sys
import os

# Add parent to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.schema_v2 import RelationalFraudDB
from privacy.masker import DataMasker

fake = Faker('en_IN')

class DemographicConfig:
    """Real-world Indian demographic distributions."""
    
    STATES_DISTRICTS = {
        'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Prayagraj', 'Agra', 'Ghaziabad'],
        'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad'],
        'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga'],
        'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri'],
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
        'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala'],
        'Delhi': ['New Delhi', 'North Delhi', 'South Delhi', 'East Delhi', 'West Delhi']
    }
    
    # Income by state (mean, std)
    INCOME_PROFILES = {
        'Delhi': (480000, 220000), 'Maharashtra': (420000, 200000),
        'Karnataka': (380000, 180000), 'Telangana': (360000, 170000),
        'Gujarat': (340000, 160000), 'Tamil Nadu': (330000, 150000),
        'Kerala': (320000, 150000), 'Haryana': (310000, 140000),
        'Punjab': (300000, 140000), 'Andhra Pradesh': (270000, 130000),
        'West Bengal': (240000, 110000), 'Rajasthan': (230000, 100000),
        'Odisha': (210000, 90000), 'Assam': (200000, 85000),
        'Jharkhand': (190000, 80000), 'Bihar': (180000, 75000),
        'Uttar Pradesh': (190000, 80000), 'Madhya Pradesh': (200000, 85000)
    }
    
    SCHEME_AMOUNTS = {
        'PDS': (1500, 3000),        # Monthly ration subsidy
        'PAHAL': (800, 1200),       # LPG cylinder subsidy
        'PM_KISAN': (6000, 6000),   # Yearly (fixed)
        'PENSION': (1000, 2500),    # Monthly pension
        'SCHOLARSHIP': (5000, 50000) # One-time/annual
    }

class RelationalDataGenerator:
    """Generate relational dataset with referential integrity."""
    
    def __init__(self, seed: int = 2024):
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)
        
        self.db = RelationalFraudDB()
        self.states = list(DemographicConfig.STATES_DISTRICTS.keys())
        
        # Tracking sets for uniqueness
        self.used_aadhaars: Set[str] = set()
        self.used_banks: Set[str] = set()
        self.used_phones: Set[str] = set()
        
        # Storage for relational linking
        self.beneficiary_ids: List[str] = []
        self.agent_ids: List[str] = []
        
    def _generate_unique_id(self, prefix: str, length: int = 8) -> str:
        """Generate unique identifier."""
        return f"{prefix}{random.randint(10**(length-1), 10**length - 1)}"
    
    def _generate_aadhaar(self) -> str:
        """Unique 12-digit Aadhaar."""
        while True:
            aadhaar = ''.join([str(random.randint(0, 9)) for _ in range(12)])
            if aadhaar not in self.used_aadhaars:
                self.used_aadhaars.add(aadhaar)
                return aadhaar
    
    def _generate_bank(self) -> Tuple[str, str]:
        """Unique bank account + IFSC."""
        banks = ['SBIN', 'HDFC', 'ICIC', 'PNB', 'BOB', 'CBI', 'UCO', 'AXIS', 'KVB', 'CANARA']
        while True:
            bank = random.choice(banks)
            branch_code = random.randint(10000, 99999)
            account = ''.join([str(random.randint(0, 9)) for _ in range(11, 16)])
            ifsc = f"{bank}0{branch_code:05d}"
            
            if account not in self.used_banks:
                self.used_banks.add(account)
                return account, ifsc
    
    def _generate_phone(self) -> str:
        """Unique 10-digit mobile."""
        while True:
            # Real Indian mobile prefixes
            prefixes = ['9', '8', '7', '6']
            prefix = random.choice(prefixes)
            rest = ''.join([str(random.randint(0, 9)) for _ in range(9)])
            phone = prefix + rest
            
            if phone not in self.used_phones:
                self.used_phones.add(phone)
                return phone
    
    def _get_income(self, state: str) -> float:
        """Realistic income based on state."""
        mean, std = DemographicConfig.INCOME_PROFILES.get(state, (250000, 100000))
        income = np.random.normal(mean, std)
        return max(20000, min(income, 10000000))  # Clamp 20K to 1Cr
    
    def generate_agents(self, count: int = 5000) -> List[Dict]:
        """Generate PDS shops, gas agencies, banks with geo-coordinates."""
        print(f"\n🏪 Generating {count:,} agents...")
        
        agents = []
        agent_types = ['pds_shop'] * 2000 + ['gas_agency'] * 1500 + \
                     ['bank_branch'] * 1000 + ['online_portal'] * 400 + ['CSC'] * 100
        
        for i in range(count):
            state = random.choice(self.states)
            district = random.choice(DemographicConfig.STATES_DISTRICTS[state])
            
            agent_id = f"AGT{i:06d}"
            
            # Generate coordinates near district center (simulated)
            base_lat, base_lon = float(fake.latitude()), float(fake.longitude())
            lat = base_lat + random.uniform(-0.5, 0.5)
            lon = base_lon + random.uniform(-0.5, 0.5)
            
            agents.append({
                'agent_id': agent_id,
                'agent_type': agent_types[i],
                'name': f"{fake.company()} {random.choice(['Store', 'Agency', 'Branch', 'Centre', 'Services'])}",
                'district': district,
                'state': state,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'license_number': f"LIC{random.randint(100000, 999999)}",
                'fraud_score': 0.0,
                'status': 'active'
            })
            
            self.agent_ids.append(agent_id)
        
        # Save to DB
        self.db.bulk_insert_agents(agents)
        return agents
    
    def generate_beneficiaries(self, count: int = 100000, fraud_rate: float = 0.30) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate 100K beneficiaries with 15% fraud injection.
        Returns: (legitimate_records, fraud_records)
        """
        print(f"\n👥 Generating {count:,} beneficiaries...")
        
        all_records = []
        fraud_records = []
        current_idx = 0
        
        fraud_total = int(count * fraud_rate)
        legit_total = count - fraud_total
        
        # Generate LEGITIMATE beneficiaries
        print("  Creating legitimate beneficiaries...")
        for i in tqdm(range(legit_total)):
            rec = self._create_beneficiary(current_idx, is_fraud=False)
            all_records.append(rec)
            current_idx += 1
        
        # Generate FRAUD patterns
        print("  Injecting fraud patterns...")
        fraud_types = [
            ('duplicate_aadhaar', int(fraud_total * 0.25)),      # 25% of fraud
            ('shared_bank', int(fraud_total * 0.20)),             # 20% - shared accounts
            ('income_mismatch', int(fraud_total * 0.20)),         # 20% - high income, low schemes
            ('ghost_identity', int(fraud_total * 0.15)),          # 15% - fake people
            ('multiple_phones', int(fraud_total * 0.10)),         # 10% - 1 person, many phones
            ('address_cluster', int(fraud_total * 0.10))          # 10% - many at 1 address
        ]
        
        for fraud_type, num_cases in fraud_types:
            cluster_size = random.randint(3, 8)
            clusters_needed = max(1, num_cases // cluster_size)
            
            for _ in range(clusters_needed):
                cluster = self._create_fraud_cluster(current_idx, fraud_type, cluster_size)
                all_records.extend(cluster)
                fraud_records.extend(cluster)
                current_idx += len(cluster)
        
        print(f"  ✓ Total generated: {len(all_records):,} (Fraud: {len(fraud_records):,})")
        return all_records, fraud_records
    
    def _create_beneficiary(self, idx: int, is_fraud: bool = False, 
                           forced_aadhaar: str = None, forced_bank: str = None,
                           forced_address: str = None, forced_phone: str = None) -> Dict:
        """Create single beneficiary record with privacy masking."""
        state = random.choice(self.states)
        district = random.choice(DemographicConfig.STATES_DISTRICTS[state])
        
        # Generate or use forced IDs
        aadhaar = forced_aadhaar or self._generate_aadhaar()
        bank_acc, ifsc = self._generate_bank() if not forced_bank else (forced_bank, "SBIN00000")
        phone = forced_phone or self._generate_phone()
        
        # Hash and mask
        aadhaar_hash = DataMasker.hash_identifier(aadhaar)
        aadhaar_masked = DataMasker.mask_aadhaar(aadhaar)
        phone_hash = DataMasker.hash_identifier(phone)
        phone_masked = DataMasker.mask_phone(phone)
        bank_hash = DataMasker.hash_identifier(bank_acc)
        bank_masked = DataMasker.mask_bank_account(bank_acc)
        
        income = self._get_income(state)
        reg_date = datetime.now() - timedelta(days=random.randint(1, 365*5))
        
        return {
            'beneficiary_id': f"BEN{idx:08d}",
            'aadhaar_hash': aadhaar_hash,
            'aadhaar_masked': aadhaar_masked,
            'name': fake.name(),
            'address': forced_address or fake.address().replace('\n', ', '),
            'phone_hash': phone_hash,
            'phone_masked': phone_masked,
            'bank_hash': bank_hash,
            'bank_masked': bank_masked,
            'ifsc_code': ifsc,
            'annual_income': round(income, 2),
            'occupation': random.choice(['Farmer', 'Laborer', 'Shopkeeper', 'Retired', 'Unemployed', 'Government', 'Private']),
            'family_size': random.randint(1, 8),
            'district': district,
            'state': state,
            'pincode': fake.postcode(),
            'registration_date': reg_date.strftime('%Y-%m-%d'),
            'status': 'active'
        }
    
    def _create_fraud_cluster(self, start_idx: int, fraud_type: str, size: int) -> List[Dict]:
        """Generate coordinated fraud ring."""
        records = []
        
        if fraud_type == 'duplicate_aadhaar':
            # Same person, multiple accounts
            master_aadhaar = self._generate_aadhaar()
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True, 
                                               forced_aadhaar=master_aadhaar)
                rec['annual_income'] = random.uniform(500000, 2000000)  # High income scam
                records.append(rec)
                
        elif fraud_type == 'shared_bank':
            # Fraud ring sharing bank account
            shared_bank, _ = self._generate_bank()
            shared_address = fake.address().replace('\n', ', ')
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True,
                                               forced_bank=shared_bank,
                                               forced_address=shared_address)
                records.append(rec)
                
        elif fraud_type == 'income_mismatch':
            # Rich people claiming poor subsidies
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True)
                rec['annual_income'] = random.uniform(1000000, 5000000)  # Rich
                records.append(rec)
                
        elif fraud_type == 'ghost_identity':
            # Fake people with suspicious patterns
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True)
                rec['name'] = f"Ghost_{random.randint(1000, 9999)}"
                rec['address'] = f"Unknown Location {i}, {rec['district']}"
                rec['annual_income'] = random.uniform(1000, 10000)
                records.append(rec)
                
        elif fraud_type == 'multiple_phones':
            # 1 person, many phone numbers (same address, similar names)
            base_address = fake.address().replace('\n', ', ')
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True,
                                               forced_address=base_address)
                records.append(rec)
                
        elif fraud_type == 'address_cluster':
            # 10+ people at same address (physically impossible)
            shared_address = fake.address().replace('\n', ', ')
            for i in range(size):
                rec = self._create_beneficiary(start_idx + i, is_fraud=True,
                                               forced_address=shared_address)
                records.append(rec)
        
        return records
    
    def generate_transactions(self, beneficiary_count: int = 100000, 
                             transactions_per_beneficiary: int = 4) -> List[Dict]:
        """
        Generate 400K transactions (4 per beneficiary on average).
        Includes temporal patterns and geo-anomalies.
        """
        total_txns = beneficiary_count * transactions_per_beneficiary
        print(f"\n💰 Generating {total_txns:,} transactions...")
        
        transactions = []
        
        # Load existing beneficiaries and agents for FK linking
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT beneficiary_id, state, district FROM beneficiaries")
            beneficiaries = cursor.fetchall()
            
            cursor.execute("SELECT agent_id, district, latitude, longitude FROM agents")
            agents = cursor.fetchall()
        
        # Group agents by district for realistic assignment
        district_agents = {}
        for agent in agents:
            dist = agent[1]
            if dist not in district_agents:
                district_agents[dist] = []
            district_agents[dist].append(agent)
        
        txn_id = 0
        for ben_id, state, district in tqdm(beneficiaries[:beneficiary_count]):
            # 3-5 transactions per beneficiary over 2 years
            num_txns = random.randint(3, 5)
            
            # Find agents in same district (80% probability) or nearby (20%)
            if district in district_agents and random.random() < 0.8:
                agent = random.choice(district_agents[district])
            else:
                agent = random.choice(agents)
            
            agent_id, agent_dist, agent_lat, agent_lon = agent
            
            # Generate transaction history
            for i in range(num_txns):
                scheme = random.choice(list(DemographicConfig.SCHEME_AMOUNTS.keys()))
                min_amt, max_amt = DemographicConfig.SCHEME_AMOUNTS[scheme]
                
                # Temporal distribution (seasonal spikes)
                days_back = random.randint(1, 730)  # 2 years
                if random.random() < 0.3:  # 30% during harvest/festival
                    days_back = random.randint(300, 400)  # Around Diwali/winter
                
                txn_date = datetime.now() - timedelta(days=days_back)
                
                # Geo-coordinate (usually near agent, sometimes anomalous)
                lat, lon = agent_lat, agent_lon
                if random.random() < 0.1:  # 10% geo-anomaly
                    lat += random.uniform(-2, 2)
                    lon += random.uniform(-2, 2)
                
                transactions.append({
                    'transaction_id': f"TXN{txn_id:010d}",
                    'beneficiary_id': ben_id,
                    'agent_id': agent_id,
                    'scheme_type': scheme,
                    'amount': round(random.uniform(min_amt, max_amt), 2),
                    'transaction_date': txn_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'channel': random.choice(['online', 'offline', 'AEPS', 'UPI', 'NEFT']),
                    'status': random.choices(['success', 'failed', 'success'], weights=[0.95, 0.03, 0.02])[0],
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'device_id': f"DEV{random.randint(1000, 9999)}" if random.random() < 0.3 else None
                })
                
                txn_id += 1
        
        # Save batch-wise
        self.db.bulk_insert_transactions(transactions)
        return transactions
    
    def run_full_generation(self):
        """Execute complete data generation pipeline."""
        print("="*60)
        print("  FRAUD DETECTION SYSTEM - DATA GENERATOR")
        print("  Target: 100K Beneficiaries | 5K Agents | 400K Transactions")
        print("="*60)
        
        # Step 1: Agents
        self.generate_agents(5000)
        
        # Step 2: Beneficiaries (with fraud injection)
        all_beneficiaries, fraud_beneficiaries = self.generate_beneficiaries(100000, fraud_rate=0.30)
        self.db.bulk_insert_beneficiaries(all_beneficiaries)
        
        # Step 3: Transactions
        self.generate_transactions(100000, 4)
        
        # Final stats
        print("\n" + "="*60)
        print("  GENERATION COMPLETE")
        print("="*60)
        stats = self.db.get_statistics()
        for key, value in stats.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        
        fraud_pct = (len(fraud_beneficiaries) / len(all_beneficiaries)) * 100
        print(f"\n  Fraud Rate: {fraud_pct:.1f}%")
        print(f"  Database File: data/processed/fraud_system.db")

if __name__ == "__main__":
    gen = RelationalDataGenerator(seed=2024)
    gen.run_full_generation()