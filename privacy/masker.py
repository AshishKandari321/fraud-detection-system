"""Privacy protection utilities for sensitive beneficiary data."""
import hashlib
import re
from typing import Dict

class DataMasker:
    """Handles PII masking and hashing for Aadhaar, Bank Accounts, and Phone numbers."""
    
    @staticmethod
    def mask_aadhaar(aadhaar: str) -> str:
        """Mask Aadhaar: XXXX-XXXX-1234"""
        if len(aadhaar) < 4:
            return "XXXX"
        return f"XXXX-XXXX-{aadhaar[-4:]}"
    
    @staticmethod
    def mask_bank_account(account: str) -> str:
        """Mask bank account: show last 4 digits only."""
        if len(account) <= 4:
            return "XXXX"
        return f"{'X' * (len(account)-4)}{account[-4:]}"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone: XXXXXX1234"""
        if len(phone) < 4:
            return "XXXX"
        return f"{'X' * (len(phone)-4)}{phone[-4:]}"
    
    @staticmethod
    def hash_identifier(identifier: str, salt: str = "fraud_detect_v1") -> str:
        """Create secure hash for deduplication without exposing raw data."""
        return hashlib.sha256(f"{identifier}{salt}".encode()).hexdigest()[:16]
    
    @staticmethod
    def apply_privacy_mask(record: Dict) -> Dict:
        """Apply all masks to a record."""
        masked = record.copy()
        if 'aadhaar_number' in masked:
            masked['aadhaar_masked'] = DataMasker.mask_aadhaar(masked['aadhaar_number'])
            del masked['aadhaar_number']
        if 'bank_account' in masked:
            masked['bank_masked'] = DataMasker.mask_bank_account(masked['bank_account'])
            del masked['bank_account']
        if 'phone_number' in masked:
            masked['phone_masked'] = DataMasker.mask_phone(masked['phone_number'])
            del masked['phone_number']
        return masked