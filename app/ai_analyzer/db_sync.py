import asyncio
import os
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
from .tools import init_db_from_dataframe

load_dotenv()

logger = logging.getLogger(__name__)

async def sync_mongodb_to_sqlite(db_path: str):
    """
    Fetches data from MongoDB 'dummy_invoice' collection in 'Fintech' database
    and syncs it to the local SQLite database.
    """
    try:
        uri = os.getenv("MONGO_URI")
        if not uri:
            logger.error("MONGO_URI not found in environment variables.")
            return False

        client = AsyncIOMotorClient(uri)
        # Explicitly use 'Fintech' database as discovered during research
        db = client["Fintech"]
        collection = db["dummy_invoice"]

        logger.info(f"Fetching data from MongoDB collection: {collection.name}")
        cursor = collection.find({})
        docs = await cursor.to_list(length=None)

        if not docs:
            logger.warning("No documents found in MongoDB collection.")
            return False

        # Flatten documents for DataFrame
        def flatten_dict(d, prefix=''):
            items = {}
            for k, v in d.items():
                new_key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key))
                elif isinstance(v, list):
                    items[new_key] = str(v)
                elif k == "_id":
                    items[new_key] = str(v)
                else:
                    items[new_key] = v
            return items

        flattened_docs = [flatten_dict(doc) for doc in docs]

        df = pd.DataFrame(flattened_docs)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        logger.info(f"Initializing SQLite database from {len(df)} documents at {db_path}")
        init_db_from_dataframe(df, db_path)
        
        logger.info("MongoDB to SQLite sync completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error during MongoDB to SQLite sync: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # For manual testing
    # Note: relative import won't work if run directly, but this is for reference
    # In production, this is called by agentMain.py
    pass
