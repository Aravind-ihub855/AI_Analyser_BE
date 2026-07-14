from fastapi import Header, HTTPException
import logging

logger = logging.getLogger(__name__)

async def get_current_user(authorization: str = Header(None)):
    """
    Placeholder: Simple middleware to extract 'user' from a token or just return a guest user.
    In production, this would verify a JWT.
    """
    # If no header, return a default system user for now to allow testing
    if not authorization:
        return {"id": "guest_id", "email": "guest@example.com", "name": "Guest User"}
    
    # Simple fake parsing: "Bearer <email>"
    try:
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            return {"id": "user_id", "email": token, "name": "Logged In User"}
    except Exception:
        pass

    return {"id": "system_user", "email": "system@example.com", "name": "System AI User"}
