from app.config.database import client
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import random
from bson import ObjectId

DB_NAME = "BP"
STORE_ID = "BP-CHI-1024"

products = client[DB_NAME]["products"]
inventory = client[DB_NAME]["inventory"]
customer_details = client[DB_NAME]["customer_details"]
purchase_history = client[DB_NAME]["purchase_history"]

engine_active = True

def _serialize(doc):
    if doc is None:
        return None
    if isinstance(doc, dict):
        return {k: _serialize(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_serialize(v) for v in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

async def get_random_customer() -> Optional[Dict]:
    pipeline = [{"$sample": {"size": 1}}]
    cursor = customer_details.aggregate(pipeline)
    result = await cursor.to_list(length=1)
    return result[0] if result else None

async def get_available_inventory(limit: int = 5) -> List[Dict]:
    cursor = inventory.find({"storeId": STORE_ID, "currentStock": {"$gt": 0}}).limit(limit * 5)
    items = await cursor.to_list(length=limit * 5)
    random.shuffle(items)
    return items[:limit]

async def create_purchase(customer: Dict, inv_items: List[Dict]) -> Optional[Dict]:
    if not engine_active:
        return None
    purchase_id = f"PUR-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
    items = []
    total = 0.0

    for inv in inv_items:
        qty = random.randint(1, 3)
        stock = int(inv.get("currentStock", 0))
        if qty > stock:
            qty = stock
        if qty <= 0:
            continue

        unit_price = float(inv.get("unitPrice", 0))
        if unit_price <= 0:
            unit_price = round(random.uniform(1.99, 25.99), 2)

        item_total = round(qty * unit_price, 2)
        total += item_total
        items.append({
            "product_code": inv.get("code", ""),
            "product_name": inv.get("name", "Unknown"),
            "category": inv.get("category", "General"),
            "quantity": qty,
            "unit_price": unit_price,
            "total_price": item_total
        })

    if not items:
        return None

    purchase = {
        "purchase_id": purchase_id,
        "store_id": STORE_ID,
        "customer_id": customer.get("customerId", ""),
        "customer_name": f"{customer.get('firstName', '')} {customer.get('lastName', '')}",
        "items": items,
        "total_amount": round(total, 2),
        "item_count": sum(i["quantity"] for i in items),
        "purchase_timestamp": datetime.utcnow(),
        "payment_method": random.choice(["Cash", "Credit Card", "Debit Card", "Mobile Pay", "Fuel Card"])
    }

    await purchase_history.insert_one(purchase)
    return purchase

async def decrement_stock(code: str, quantity: int) -> int:
    if not engine_active:
        return 0
    result = await inventory.find_one_and_update(
        {"storeId": STORE_ID, "code": code, "currentStock": {"$gte": quantity}},
        {"$inc": {"currentStock": -quantity}},
        return_document=True
    )
    return int(result.get("currentStock", 0)) if result else 0

async def check_and_restock(code: str) -> Optional[Dict]:
    inv = await inventory.find_one({"storeId": STORE_ID, "code": code})
    if not inv:
        return None

    stock = int(inv.get("currentStock", 0))
    rol = int(inv.get("rol", 20))
    roq = int(inv.get("roq", 100))

    if stock <= rol and roq > 0:
        await inventory.update_one(
            {"storeId": STORE_ID, "code": code},
            {"$inc": {"currentStock": roq}}
        )
        return {
            "product_code": code,
            "product_name": inv.get("name", ""),
            "previous_stock": stock,
            "restocked_quantity": roq,
            "new_stock": stock + roq
        }
    return None

async def get_kpis() -> Dict:
    pipeline = [
        {"$group": {
            "_id": None,
            "total_revenue": {"$sum": "$total_amount"},
            "total_orders": {"$sum": 1},
            "avg_order_value": {"$avg": "$total_amount"}
        }}
    ]
    cursor = purchase_history.aggregate(pipeline)
    result = await cursor.to_list(length=1)

    low_stock = await inventory.count_documents({
        "storeId": STORE_ID,
        "$expr": {"$lte": ["$currentStock", "$rol"]}
    })

    return {
        "total_revenue": round(result[0]["total_revenue"], 2) if result else 0,
        "total_orders": result[0]["total_orders"] if result else 0,
        "avg_order_value": round(result[0]["avg_order_value"], 2) if result else 0,
        "low_stock_products": low_stock,
        "active_customers": await customer_details.count_documents({}),
        "total_products": await inventory.count_documents({"storeId": STORE_ID})
    }

async def get_revenue_series(minutes: int = 60) -> List[Dict]:
    start = datetime.utcnow() - timedelta(minutes=minutes)
    pipeline = [
        {"$match": {"purchase_timestamp": {"$gte": start}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d %H:%M", "date": "$purchase_timestamp"}},
            "revenue": {"$sum": "$total_amount"},
            "orders": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    return await purchase_history.aggregate(pipeline).to_list(minutes + 10)

async def get_top_products(limit: int = 20) -> List[Dict]:
    pipeline = [
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.product_code",
            "product_name": {"$first": "$items.product_name"},
            "category": {"$first": "$items.category"},
            "total_quantity": {"$sum": "$items.quantity"},
            "total_revenue": {"$sum": "$items.total_price"}
        }},
        {"$sort": {"total_revenue": -1}},
        {"$limit": limit}
    ]
    return await purchase_history.aggregate(pipeline).to_list(limit)

async def get_inventory_snapshot(limit: int = 50) -> List[Dict]:
    cursor = inventory.find({"storeId": STORE_ID}).sort("currentStock", 1).limit(limit)
    raw = await cursor.to_list(length=limit)
    return [_serialize(r) for r in raw]

async def get_recent_purchases(limit: int = 50) -> List[Dict]:
    cursor = purchase_history.find().sort("purchase_timestamp", -1).limit(limit)
    raw = await cursor.to_list(length=limit)
    return [_serialize(r) for r in raw]
