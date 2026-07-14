from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
from app.streaming.websocket import ws_manager
from app.streaming.engine import streaming_engine
from app.streaming.models import (
    get_kpis, get_revenue_series, get_top_products,
    get_inventory_snapshot
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/streaming", tags=["Streaming Dashboard"])

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        kpis = await get_kpis()
        await ws_manager.send_personal({"type": "kpi_update", "data": kpis}, websocket)
        
        # Send initial data
        engine_stats = streaming_engine.get_stats()
        await ws_manager.send_personal({"type": "engine_stats", "data": engine_stats}, websocket)
        
        revenue_series = await get_revenue_series(60)
        await ws_manager.send_personal({"type": "revenue_series", "data": revenue_series}, websocket)
        
        top_products = await get_top_products(20)
        await ws_manager.send_personal({"type": "top_products", "data": top_products}, websocket)
        
        inventory = await get_inventory_snapshot(50)
        await ws_manager.send_personal({"type": "inventory_snapshot", "data": inventory}, websocket)
        
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await ws_manager.send_personal({"type": "pong"}, websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

@router.get("/kpis")
async def get_kpis_api():
    return await get_kpis()

@router.get("/revenue/series")
async def get_revenue_series_api(minutes: int = Query(60, le=1440)):
    return await get_revenue_series(minutes)

@router.get("/products/top")
async def get_top_products_api(limit: int = Query(20, le=100)):
    return await get_top_products(limit)

@router.get("/inventory/snapshot")
async def get_inventory_snapshot_api(limit: int = Query(50, le=200)):
    return await get_inventory_snapshot(limit)

@router.get("/engine/stats")
async def get_engine_stats():
    return streaming_engine.get_stats()

@router.post("/engine/start")
async def start_engine(tx_per_second: float = Query(0.083, ge=0.01, le=50.0)):
    streaming_engine.tx_per_second = tx_per_second
    streaming_engine.interval = 1.0 / tx_per_second
    await streaming_engine.start()
    await ws_manager.broadcast({"type": "engine_stats", "data": streaming_engine.get_stats()})
    return {"status": "started", "tx_per_second": tx_per_second}

@router.post("/engine/stop")
async def stop_engine():
    await streaming_engine.stop()
    await ws_manager.broadcast({"type": "engine_stats", "data": streaming_engine.get_stats()})
    return {"status": "stopped"}