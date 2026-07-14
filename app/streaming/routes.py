from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.streaming.engine import engine
from app.streaming.websocket import manager
from app.streaming.models import (
    get_kpis, get_recent_purchases, get_top_products,
    get_inventory_snapshot, get_revenue_series, get_low_stock_count
)
from app.shared.authMiddleware import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/streaming", tags=["Streaming"])


@router.on_event("startup")
async def startup_streaming():
    await engine.start()


@router.on_event("shutdown")
async def shutdown_streaming():
    await engine.stop()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        kpis = await get_kpis()
        kpis["low_stock_count"] = await get_low_stock_count()
        await manager.broadcast_kpi_update(kpis)

        recent = await get_recent_purchases(20)
        await websocket.send_text(f'{{"event": "initial_purchases", "data": {recent}}}')

        top_products = await get_top_products(10)
        await websocket.send_text(f'{{"event": "top_products", "data": {top_products}}}')

        inventory = await get_inventory_snapshot(20)
        await websocket.send_text(f'{{"event": "inventory_snapshot", "data": {inventory}}}')

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/status")
async def streaming_status(current_user: dict = None):
    return engine.get_stats()


@router.post("/start")
async def start_streaming(current_user: dict = None):
    if not engine.running:
        await engine.start()
    return {"status": "started", "stats": engine.get_stats()}


@router.post("/stop")
async def stop_streaming(current_user: dict = None):
    if engine.running:
        await engine.stop()
    return {"status": "stopped", "stats": engine.get_stats()}


@router.get("/kpis")
async def get_streaming_kpis(current_user: dict = None):
    kpis = await get_kpis()
    kpis["low_stock_count"] = await get_low_stock_count()
    kpis["engine_stats"] = engine.get_stats()
    return kpis


@router.get("/purchases/recent")
async def get_recent_purchases_api(limit: int = Query(50, le=200), current_user: dict = None):
    return await get_recent_purchases(limit)


@router.get("/products/top")
async def get_top_products_api(limit: int = Query(20, le=100), current_user: dict = None):
    return await get_top_products(limit)


@router.get("/inventory/snapshot")
async def get_inventory_snapshot_api(limit: int = Query(50, le=200), current_user: dict = None):
    return await get_inventory_snapshot(limit)


@router.get("/revenue/series")
async def get_revenue_series_api(minutes: int = Query(60, le=1440), current_user: dict = None):
    return await get_revenue_series(minutes)


@router.get("/alerts/low-stock")
async def get_low_stock_alerts(limit: int = Query(50, le=200), current_user: dict = None):
    from app.config.database import client
    collection = client["BP"]["products_stream"]
    cursor = collection.find(
        {"$expr": {"$lt": ["$stock_quantity", "$min_threshold"]}}
    ).sort("stock_quantity", 1).limit(limit)
    return await cursor.to_list(length=limit)