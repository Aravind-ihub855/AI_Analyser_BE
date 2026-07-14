from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.auth.auth_db import create_user, get_user_by_email, verify_password, generate_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def signup(req: SignupRequest):
    if not req.email or not req.password or not req.name:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
        
    user = await create_user(req.email, req.password, req.name)
    if not user:
        raise HTTPException(status_code=400, detail="User with this email already exists")
        
    # Generate token
    token = generate_token({
        "id": user["id"], 
        "email": user["email"], 
        "name": user["name"],
        "role": user.get("role", "store manager"),
        "storeId": user.get("storeId"),
        "region": user.get("region")
    })
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user.get("role", "store manager"),
            "storeId": user.get("storeId"),
            "region": user.get("region")
        }
    }

@router.post("/login")
async def login(req: LoginRequest):
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Missing email or password")
        
    user = await get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    if not verify_password(user["password"], req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    # Generate token
    token = generate_token({
        "id": user["id"], 
        "email": user["email"], 
        "name": user["name"],
        "role": user.get("role", "store manager"),
        "storeId": user.get("storeId"),
        "region": user.get("region")
    })
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user.get("role", "store manager"),
            "storeId": user.get("storeId"),
            "region": user.get("region")
        }
    }
