from app.config.database import client
from bson import ObjectId
import datetime

# Database and Collection for Chat History
DB_NAME = "BP"
COLLECTION_NAME = "chat_history"
history_collection = client[DB_NAME][COLLECTION_NAME]

async def create_new_chat(user_id: str = "SNS Ihub"):
    """Creates a new chat session in MongoDB."""
    new_chat = {
        "user_id": user_id,
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow()
    }
    result = await history_collection.insert_one(new_chat)
    return str(result.inserted_id)

async def get_user_chats(user_id: str = "SNS Ihub"):
    """Retrieves all chat sessions for a user, sorted by update date."""
    cursor = history_collection.find({"user_id": user_id}).sort("updated_at", -1)
    chats = await cursor.to_list(length=100)
    return [{"id": str(c["_id"]), "title": c["title"], "updated_at": c["updated_at"]} for c in chats]

async def get_chat_details(chat_id: str):
    """Retrieves the full message history for a specific chat ID."""
    chat = await history_collection.find_one({"_id": ObjectId(chat_id)})
    if chat:
        chat["id"] = str(chat["_id"])
        del chat["_id"]
        return chat
    return None

async def update_chat_title(chat_id: str, new_title: str):
    """Updates the display title of a chat session."""
    await history_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {"title": new_title, "updated_at": datetime.datetime.utcnow()}}
    )
    return True

async def save_chat_message(chat_id: str, role: str, content: str, tableData=None, chartData=None, reportData=None, dashboardData=None):
    """Appends a new message to the chat session."""
    message_obj = {
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.utcnow()
    }
    # Include metadata if provided (for bot responses)
    if tableData: message_obj["tableData"] = tableData
    if chartData: message_obj["chartData"] = chartData
    if reportData: message_obj["reportData"] = reportData
    if dashboardData: message_obj["dashboardData"] = dashboardData

    await history_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {
            "$push": {"messages": message_obj},
            "$set": {"updated_at": datetime.datetime.utcnow()}
        }
    )
    return True

async def delete_chat_session(chat_id: str):
    """Permanently deletes a chat session."""
    await history_collection.delete_one({"_id": ObjectId(chat_id)})
    return True
