from cryptography.fernet import Fernet
import base64
import os
from dotenv import load_dotenv
import json

load_dotenv()


class EncryptionHandler:
    """Handles encryption/decryption of sensitive data"""

    def __init__(self):
        self.cipher_suite = self._get_cipher()

    def _get_cipher(self):
        """Get cipher from environment or generate new key"""
        key = os.getenv("ENCRYPTION_KEY")

        if not key:
            # Generate new key if not in .env
            key = Fernet.generate_key().decode()
            print(f"⚠️  Generated new encryption key. Add this to .env:")
            print(f"ENCRYPTION_KEY={key}")

        if isinstance(key, str):
            key = key.encode()

        return Fernet(key)

    def encrypt(self, data):
        """
        Encrypt data (string or dict)

        Returns encrypted bytes
        """
        try:
            if isinstance(data, dict):
                data = json.dumps(data)

            if isinstance(data, str):
                data = data.encode()

            encrypted = self.cipher_suite.encrypt(data)
            return encrypted
        except Exception as e:
            print(f"✗ Encryption error: {e}")
            return None

    def decrypt(self, encrypted_data):
        """
        Decrypt data

        Returns decrypted string or dict
        """
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data)

            try:
                return json.loads(decrypted.decode())
            except json.JSONDecodeError:
                return decrypted.decode()
        except Exception as e:
            print(f"✗ Decryption error: {e}")
            return None

    def encrypt_to_string(self, data):
        """Encrypt and return as base64 string for storage"""
        encrypted = self.encrypt(data)
        if encrypted:
            return base64.b64encode(encrypted).decode()
        return None

    def decrypt_from_string(self, encrypted_string):
        """Decrypt from base64 string"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_string)
            return self.decrypt(encrypted_bytes)
        except Exception as e:
            print(f"✗ Decryption error: {e}")
            return None
