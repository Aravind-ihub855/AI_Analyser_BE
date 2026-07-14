import asyncio
import os
import sqlite3
import pandas as pd
import json
import random
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Remote MongoDB configuration
REMOTE_MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "BP"

def clean_column_name(col: str) -> str:
    """Normalize column names for SQLite safety."""
    import re
    col = col.strip()
    # Handle CamelCase: convert to snake_case
    col = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col)
    col = re.sub('([a-z0-9])([A-Z])', r'\1_\2', col).lower()
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)    # replace spaces & symbols with _
    col = re.sub(r"_+", "_", col)               # collapse __ → _
    col = col.strip("_")                        # remove leading/trailing _
    return col

def generate_mock_sales(inventory_docs, conn):
    """
    Generates realistic daily sales transactions matching
    each inventory item's unitPrice and avgDailyConsumption.
    """
    logger.info("Generating mock sales transactions for the last 7 days...")
    sales_records = []
    
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=7)
    
    transaction_counter = 100000
    
    for inv in inventory_docs:
        store_id = inv.get("storeId", "BP-CHI-1024")
        code = inv.get("code")
        name = inv.get("name")
        category = inv.get("category", "General")
        price = inv.get("unitPrice", 0.0)
        
        # Retrieve average daily consumption (fallback to random standard values if missing)
        avg_daily = inv.get("avgDailyConsumption", 0.0)
        if not avg_daily or avg_daily <= 0:
            avg_daily = random.choice([5.0, 10.0, 15.0, 20.0])
            
        # Target total sales over 7 days
        total_sales_qty = int(random.normalvariate(avg_daily * 7, avg_daily * 1.5))
        total_sales_qty = max(0, total_sales_qty)
        
        current_qty = 0
        while current_qty < total_sales_qty:
            if category.lower() == 'fuel' or category.lower() == 'automotive' and 'litres' in str(inv.get('uom','')).lower():
                qty = round(random.uniform(20.0, 60.0), 2)
            else:
                qty = random.choice([1, 1, 1, 2, 2, 3, 4, 5])
                
            current_qty += qty
            
            # Random timestamp over the last 7 days
            random_seconds = random.randint(0, 7 * 86400)
            tx_time = start_date + datetime.timedelta(seconds=random_seconds)
            
            transaction_counter += 1
            sales_records.append({
                "transaction_id": f"TX-{transaction_counter}",
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "store_id": store_id,
                "code": code,
                "name": name,
                "category": category,
                "quantity": qty,
                "price": price,
                "total_amount": round(qty * price, 2),
                "payment_method": random.choice(["Card", "Card", "Fuel Card", "Mobile", "Cash"])
            })
            
    if sales_records:
        sales_df = pd.DataFrame(sales_records)
        sales_df.to_sql('sales', conn, if_exists='replace', index=False)
        
        # Create indexes
        cursor = conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_code ON sales(code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_timestamp ON sales(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sales_store_id ON sales(store_id)')
        conn.commit()
        logger.info(f"Generated {len(sales_records)} mock sales records successfully.")

async def sync_mongodb_to_sqlite(db_path: str):
    """
    Syncs the remote BP MongoDB database to a local SQLite database for real-time querying.
    Also injects mock sales data to support sales analysis.
    """
    try:
        logger.info(f"Connecting to remote MongoDB: {REMOTE_MONGO_URI}")
        client = AsyncIOMotorClient(REMOTE_MONGO_URI)
        db = client[DB_NAME]
        
        # Ensure target SQLite directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        
        collections = ['products', 'purchase_orders', 'vendors', 'stores', 'inventory', 'vendor_issues', 'recommendations']
        
        inventory_docs = []
        
        for col_name in collections:
            logger.info(f"Syncing collection: {col_name}...")
            collection = db[col_name]
            cursor = collection.find({})
            docs = await cursor.to_list(length=None)
            
            if not docs:
                logger.warning(f"No documents found in collection: {col_name}")
                continue
                
            if col_name == 'inventory':
                inventory_docs = docs
                
            # Flatten and serialize complex columns
            flattened = []
            for doc in docs:
                item = {}
                for k, v in doc.items():
                    if k == '_id':
                        item[clean_column_name(k)] = str(v)
                    elif isinstance(v, (dict, list)):
                        # Serialize lists and sub-dicts (like PO items or store product maps)
                        item[clean_column_name(k)] = json.dumps(v)
                    else:
                        item[clean_column_name(k)] = v
                flattened.append(item)
                
            df = pd.DataFrame(flattened)
            
            # Normalise columns
            df.columns = [clean_column_name(c) for c in df.columns]
            
            # Save to SQLite table
            table_name = col_name
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Saved {len(df)} rows to table '{table_name}'.")
            
            # Create indexing on standard IDs and Codes
            cursor = conn.cursor()
            for col in df.columns:
                if col in ['code', 'id', 'sku', 'store_id', 'vendor_id']:
                    try:
                        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON "{table_name}"("{col}")')
                    except Exception:
                        pass
            conn.commit()
            
        # Generate mock sales
        generate_mock_sales(inventory_docs, conn)
        
        conn.close()
        client.close()
        logger.info("Database sync and mock sales generation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during MongoDB to SQLite sync: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Test script running
    logging.basicConfig(level=logging.INFO)
    db_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "store_ops.db")
    asyncio.run(sync_mongodb_to_sqlite(db_file_path))
