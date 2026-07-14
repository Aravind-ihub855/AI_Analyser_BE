import asyncio
import logging
import random
import os
from typing import Optional
from app.streaming.models import (
    get_random_customer, get_available_inventory, create_purchase, get_kpis
)
from app.streaming.websocket import ws_manager

logger = logging.getLogger(__name__)

TX_PER_SECOND = float(os.getenv("STREAM_TX_PER_SECOND", "0.083"))
PEAK_MULTIPLIER = float(os.getenv("STREAM_PEAK_MULTIPLIER", "2.0"))

class StreamingEngine:
    def __init__(self, tx_per_second: float = TX_PER_SECOND):
        self.tx_per_second = tx_per_second
        self.interval = 1.0 / tx_per_second
        self.running = False
        self._stop_event = asyncio.Event()
        self._stop_event.set()
        self.task: Optional[asyncio.Task] = None
        self.stats = {"purchases": 0, "errors": 0}

    async def start(self):
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self.stats = {"purchases": 0, "errors": 0}
        import app.streaming.models as m
        m.engine_active = True
        self.task = asyncio.create_task(self._run_loop())
        logger.info(f"Streaming engine started at {self.tx_per_second} tx/sec")

    async def stop(self):
        self.running = False
        self._stop_event.set()
        import app.streaming.models as m
        m.engine_active = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        logger.info(f"Streaming engine stopped. Stats: {self.stats}")

    def _current_rate(self) -> float:
        from datetime import datetime
        hour = datetime.utcnow().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return self.tx_per_second * PEAK_MULTIPLIER
        return self.tx_per_second

    async def _run_loop(self):
        while self.running and not self._stop_event.is_set():
            start = asyncio.get_event_loop().time()
            try:
                await self._process_transaction()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Transaction error: {e}")

            elapsed = asyncio.get_event_loop().time() - start
            base_interval = 1.0 / self._current_rate()
            jitter = random.uniform(-0.5, 0.5) * base_interval
            sleep_time = max(0, base_interval + jitter - elapsed)
            if sleep_time > 0:
                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

    async def _process_transaction(self):
        if not self.running or self._stop_event.is_set():
            return

        customer = await get_random_customer()
        if not customer:
            return

        inv_items = await get_available_inventory(limit=random.randint(1, 5))
        if not inv_items:
            return

        purchase = await create_purchase(customer, inv_items)
        if not purchase:
            return

        self.stats["purchases"] += 1

        await ws_manager.broadcast({
            "type": "purchase_created",
            "data": {
                "purchase_id": purchase["purchase_id"],
                "customer": purchase["customer_name"],
                "items": purchase["items"],
                "total": purchase["total_amount"],
                "timestamp": purchase["purchase_timestamp"].isoformat() + "Z",
                "payment_method": purchase["payment_method"]
            }
        })

        kpis = await get_kpis()
        await ws_manager.broadcast({"type": "kpi_update", "data": kpis})

        engine_stats = self.get_stats()
        await ws_manager.broadcast({"type": "engine_stats", "data": engine_stats})

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "running": self.running,
            "current_rate": self._current_rate()
        }

streaming_engine = StreamingEngine()
