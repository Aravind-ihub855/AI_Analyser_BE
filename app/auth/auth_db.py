import hashlib
import os
import secrets
import base64
import json
import time
import datetime
import hmac
from bson import ObjectId
from app.config.database import client

DB_NAME = "Fintech_AI_Analyser"
COLLECTION_NAME = "users"
users_collection = client[DB_NAME][COLLECTION_NAME]

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-me-1234567890")

# --- HASHING UTILS ---
def hash_password(password: str) -> str:
    # Generate a random 16-byte salt
    salt = os.urandom(16)
    # Perform 100,000 iterations of PBKDF2 HMAC SHA-256
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    # Store both salt and hash as hex
    return f"{salt.hex()}:{pwd_hash.hex()}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    try:
        salt_hex, hash_hex = stored_password.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(expected_hash, pwd_hash)
    except Exception:
        return False

# --- TOKEN UTILS ---
def generate_token(payload: dict, expires_in: int = 86400) -> str:
    payload_copy = payload.copy()
    payload_copy["exp"] = int(time.time()) + expires_in
    
    # Base64 encode the payload
    payload_json = json.dumps(payload_copy).encode('utf-8')
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode('utf-8').rstrip('=')
    
    # Calculate HMAC signature
    sig = hmac_new_signature(payload_b64)
    return f"{payload_b64}.{sig}"

def verify_token(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        
        # Verify signature first
        expected_sig = hmac_new_signature(payload_b64)
        if not secrets.compare_digest(expected_sig, sig_b64):
            return None
            
        # Decode payload
        rem = len(payload_b64) % 4
        if rem > 0:
            payload_b64 += "=" * (4 - rem)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
        payload = json.loads(payload_json)
        
        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None
            
        return payload
    except Exception:
        return None

def hmac_new_signature(payload_b64: str) -> str:
    sig = hmac.new(SECRET_KEY.encode('utf-8'), payload_b64.encode('utf-8'), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode('utf-8').rstrip('=')

# --- DB HELPERS ---
async def create_user(email: str, password_raw: str, name: str) -> dict:
    existing = await users_collection.find_one({"email": email.lower()})
    if existing:
        return None
        
    hashed_pwd = hash_password(password_raw)
    new_user = {
        "email": email.lower(),
        "password": hashed_pwd,
        "name": name,
        "created_at": datetime.datetime.utcnow()
    }
    result = await users_collection.insert_one(new_user)
    new_user["id"] = str(result.inserted_id)
    return new_user

async def get_user_by_email(email: str) -> dict:
    user = await users_collection.find_one({"email": email.lower()})
    if user:
        user["id"] = str(user["_id"])
        return user
    return None

async def get_user_by_id(user_id: str) -> dict:
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["id"] = str(user["_id"])
            return user
    except Exception:
        pass
    return None
