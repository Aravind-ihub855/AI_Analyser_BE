from fastapi import WebSocket
from typing import Set, Dict, Any
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = set()
        
        async with self._lock:
            for ws in self.active_connections:
                try:
                    await ws.send_text(message_str)
                except Exception as e:
                    logger.warning(f"WebSocket send failed: {e}")
                    disconnected.add(ws)
            
            for ws in disconnected:
                self.active_connections.discard(ws)
    
    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.warning(f"Personal message failed: {e}")
            self.disconnect(websocket)

ws_manager = WebSocketManager()