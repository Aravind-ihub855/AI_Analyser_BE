from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from .agents import parse_query_to_tool
from .tools import get_data_summary, detect_anomalies, generate_insights, query_dataset, generate_detailed_report, generate_visualization, init_db_from_dataframe, query_salesforce_customer
from .date_converter import convert_excel_dates_in_dataframe
from .db_sync import sync_mongodb_to_sqlite
import shutil
import os
import tempfile
from typing import List, Tuple, Dict
import json
import pandas as pd
import sqlite3
import logging
from dotenv import load_dotenv

from app.services.llm import get_groq_llm
from app.shared.authMiddleware import get_current_user
from app.services.user_usage import (
    increment_user_used_tokens,
    ensure_usage_quota_available,
)
from .agent_orchestrator import AgentOrchestrator

load_dotenv()
llm = get_groq_llm()
orchestrator = AgentOrchestrator()

router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "store_ops.db") # Use store ops DB name

# Global variable to track current DB path
current_db_path = DB_PATH
# Note: Global conversation_history is not thread-safe/user-safe. 
# Ideally, history should be passed from frontend or stored per-session.
# We will rely on the frontend passing the history in the 'conversation' parameter.
conversation_history = []

def _extract_token_usage(response) -> Dict[str, int]:
    """Extract token usage from LLM response."""
    usage = getattr(response, "usage_metadata", {}) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0)
    }

def _accumulate_usage(total_usage: Dict[str, int], new_usage: Dict[str, int]) -> None:
    """Accumulate token usage into total_usage in-place."""
    total_usage["input_tokens"] += new_usage.get("input_tokens", 0)
    total_usage["output_tokens"] += new_usage.get("output_tokens", 0)
    total_usage["total_tokens"] += new_usage.get("total_tokens", 0)

def _get_dataset_preview() -> List[List]:
    """Helper: preview from DB."""
    global current_db_path
    try:
        if not current_db_path or not os.path.exists(current_db_path):
            return [["Status"], ["No data available. Synchronization might be in progress."]]
        
        conn = sqlite3.connect(current_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(inventory)")
        columns = [col[1] for col in cursor.fetchall()]
        df = pd.read_sql_query("SELECT * FROM inventory LIMIT 20", conn)
        conn.close()
        
        # Replace NaN/inf with None for JSON compliance
        # astype(object) is crucial to allow None in numeric columns
        df = df.astype(object).where(pd.notnull(df), None)
        
        # Prepare list-of-lists format for frontend
        return [columns] + df.values.tolist()
    except Exception as e:
        logger.error(f"Error loading preview: {str(e)}")
        return [["Error"], [str(e)]]

@router.on_event("startup")
async def startup_event():
    """Trigger initial data sync on startup."""
    logger.info("Starting initial MongoDB to SQLite sync...")
    await sync_mongodb_to_sqlite(DB_PATH)

@router.get("/get-preview")
async def get_preview():
    """Fetch the current data preview for the frontend."""
    preview_data = _get_dataset_preview()
    return {"data": preview_data}

async def generate_acknowledgement_message(tool_name: str, llm) -> Tuple[str, Dict[str, int]]:
    """
    Dynamically generates a catchy AI acknowledgment message after completing an action.
    No predefined wording — AI creates a short, human-like confirmation (1-2 sentences).
    """
    prompt = f"""
    You are the BP Store Manager AI Copilot, an intelligent and friendly store operations assistant.

    The user has just received the output for a tool operation: "{tool_name}".

    Generate a short, engaging confirmation message (1-2 lines max) acknowledging 
    that the requested action is complete and the results are visible on the right panel.

    Requirements:
    - Be natural, varied, and human-like — never robotic or repetitive.
    - Avoid generic templates like "I've generated…" or "Here's your…"
    - You can use expressions like "All set!", "Ready when you are!", "Your chart's shining bright!", etc.
    - Never mention technical words like SQL, dataset, or backend.
    - Always sound like a Financial Intelligence officer or Advisor.
    - Keep it concise, conversational, and professional.
    - **Markdown is ALLOWED** (e.g., use bold for emphasis) but keep it to 1-2 lines.
    - Return ONLY the message.

    Example context:
    - tool_name: generate_visualization → Could be "Your chart's ready! Take a look on the right panel."
    - tool_name: generate_detailed_report → Could be "Your report's ready with deep insights — check the right panel!"

    Now, generate the final acknowledgment line for:
    Tool: {tool_name}
    """
    try:
        response = await llm.ainvoke(prompt)
        usage = _extract_token_usage(response)
        message = response.content.strip()
        # Clean up unexpected newlines or extra text
        final_msg = message.splitlines()[0].strip() if message else "All done! You can check it on the right panel."
        return final_msg, usage
    except Exception as e:
        logger.warning(f"AI acknowledgment generation failed: {str(e)}")
        return "Your result is ready — check it out on the right panel!", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


async def process_query(query: str, conversation: List[str], current_user: dict = None):
    global current_db_path, conversation_history
    
    usage_record = None
    # Ensure usage quota is available if user is authenticated
    if current_user:
        usage_record = ensure_usage_quota_available(current_user)

    if not current_db_path or not os.path.exists(current_db_path):
        logger.error("Query attempted but no database found.")
        return {
            "response": "Error: Data source is not available. Please wait for synchronization.",
            "is_report": False, "visualization": None,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0
        }
    
    db_path = current_db_path
    
    # Update conversation history with provided conversation
    # We use a local variable for the current request's context to avoid global state issues if possible,
    # but we also update the global one for fallback.
    current_conversation_history = []
    
    if conversation:
        try:
            # Parse conversation from frontend
            parsed_history = []
            for conv_str in conversation:
                msg = json.loads(conv_str)
                parsed_history.append({"user": msg.get("user"), "ai": msg.get("ai")})
            
            # Robust History Management: Keep at least 40 messages to retain 15-20 turns of memory context
            if len(parsed_history) > 40:
                parsed_history = parsed_history[-40:]
            
            current_conversation_history = parsed_history
            conversation_history = parsed_history # Sync global (though global is not ideal for multi-user)
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid conversation string received from frontend.")
            current_conversation_history = conversation_history[-40:] if len(conversation_history) > 40 else conversation_history
    else:
        # Fallback to global history if not provided
        current_conversation_history = conversation_history[-40:] if len(conversation_history) > 40 else conversation_history

    # Parse query to determine the tool or conversational response
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        logger.debug(f"Parsing query: {query}")
        # Heuristic: Treat common greeting/help intents as conversational
        q_lower = (query or "").strip().lower()
        greeting_intents = [
            "hi", "hello", "hey", "who are you", "who are u", "how can you help", "how can u help",
            "what can you do", "help", "can you help", "how do you work"
        ]
        is_greeting = any(q_lower == phrase for phrase in greeting_intents)

        if is_greeting:
            tool_response = "conversational"
        else:
            tool_response, router_usage = await parse_query_to_tool(llm, query, current_conversation_history, current_db_path)
            _accumulate_usage(total_usage, router_usage)
        logger.debug(f"Parse result: {tool_response}")
    except Exception as e:
        logger.error(f"Error parsing query: {str(e)}")
        error_str = str(e)
        friendly_error = f"Error: Failed to parse query '{query}'. Please try again."
        if "429" in error_str or "rate limit" in error_str.lower():
            friendly_error = "I'm currently experiencing a high volume of requests. Please try again in a moment, and I'll be ready to assist you!"
            
        return {
            "response": friendly_error,
            "summary": "",
            "anomalies": "",
            "insights": "",
            "is_report": False,
            "visualization": None,
            "total_input_tokens": total_usage["input_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"]
        }

    # Handle clarification request
    if tool_response.startswith("Could you clarify"):
        # We don't append to history here because the frontend will do it when it receives the response
        return {
            "response": tool_response,
            "summary": "",
            "anomalies": "",
            "insights": "",
            "is_report": False,
            "visualization": None,
            "total_input_tokens": total_usage["input_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"]
        }

    if tool_response == "conversational":
        # Generate a general conversational response using LLM (not SQL)
        history_text = "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in current_conversation_history])
        prompt = f"""
        You are **BP Store Manager AI Copilot** — a conversational, multi-level AI Agentic Copilot primarily focused on supporting BP store managers, operations advisors, and vendor managers.

        ### Your Role & Identity:
        - You are the BP Store Manager AI Copilot, capable of understanding operational questions, analyzing live inventory, accessing enterprise knowledge, executing workflows, monitoring events, and providing recommendations in real time.
        - You are an Agentic AI System, not a normal chatbot. This means you do not only answer questions, but you can also: Answer, Analyze, Compare, Predict, Recommend, Execute, Monitor, and Notify.
        - You communicate in a professional, warm, and operations-aware tone.

        ### Your Capabilities:
        - You assist with database stats, inventory checks, RAG guidelines, local events, FRED market trends, competitor pricing, weather tracking, and Salesforce customer database queries.
        - You can also engage in friendly chats, explain features of the Copilot, or guide users on how to use operational tools.
        - **CRITICAL**: Do NOT output Python code. Provide operational/financial results instead.
        - You do **not** execute or reference SQL in general or casual chats.

        ### Your Behaviour Rules:
        1. If the user greets you, compliments you, or talks casually — respond naturally in a friendly, concise tone.
        2. If the query mentions your abilities, explain them briefly and helpfully.
        3. Never mention technical back-end details like SQL or databases unless explicitly asked.
        4. Keep responses short (1–3 sentences), polite, and contextually relevant.
        5. Sound human and conversational — not robotic or salesy.

        ---

        **Conversation so far:**
        {history_text}

        **User Query:** {query}

        Respond naturally in conversational tone — warm, informative, and aligned with AI Sheet's identity. 
        Avoid any data, code, or SQL references.
        """

        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        response = response_obj.content.strip()
        # logger.info(f"Conversational response: {response}")
        
        # Increment user tokens if authenticated
        if current_user:
            increment_user_used_tokens(
                current_user,
                total_usage["total_tokens"],
                usage_record=usage_record,
            )
        
        return {
            "response": response,
            "summary": "",
            "anomalies": "",
            "insights": "",
            "is_report": False,
            "visualization": None,
            "total_input_tokens": total_usage["input_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"]
        }

    # Map tool names to functions (updated to use db_path)
    tool_map = {
        "get_data_summary": lambda q, c: get_data_summary(db_path, q, c),
        "detect_anomalies": lambda q, c: detect_anomalies(db_path, q, c),
        "generate_insights": lambda q, c: generate_insights(db_path, q, c),
        "query_dataset": lambda q, c: query_dataset(db_path, q, c),
        "generate_detailed_report": lambda q, c: generate_detailed_report(db_path, q, c),
        "generate_visualization": lambda q, c: generate_visualization(db_path, q, c),
        "query_salesforce_customer": lambda q, c: query_salesforce_customer(q, c)
    }

    if tool_response not in tool_map:
        logger.error(f"Invalid tool selected: {tool_response}")
        # Fallback to query_dataset if an invalid tool is somehow chosen for a data query
        tool_func = tool_map["query_dataset"]
        response, tool_usage = await tool_func(query, current_conversation_history)
        _accumulate_usage(total_usage, tool_usage)
        logger.warning(f"Invalid tool '{tool_response}' selected, falling back to query_dataset.")
    else:
        # Call the selected tool
        tool_func = tool_map[tool_response]
        response, tool_usage = await tool_func(query, current_conversation_history)
        _accumulate_usage(total_usage, tool_usage)
    
    try:
        logger.debug(f"Calling tool: {tool_response}")
        # logger.info(f"Tool used: {tool_response}\nTool output: {response}")

        # Handle visualization response (JSON)
        visualization = None
        if tool_response == "generate_visualization":
            try:
                visualization = json.loads(response)
                response = visualization.get("explanation", response)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse visualization response as JSON: {response}")
                visualization = None

        # Determine if the response is a report or visualization
        is_report = (tool_response == "generate_detailed_report")
        is_visualization = (tool_response == "generate_visualization")

        # 🎨 NEW ADDITION: AI-generated acknowledgment message
        ack_message = None
        if tool_response in ["generate_detailed_report", "generate_visualization"]:
            ack_message, ack_usage = await generate_acknowledgement_message(tool_response, llm)
            _accumulate_usage(total_usage, ack_usage)
            # logger.info(f"Acknowledgement message generated: {ack_message}")

        # Compose final message
        final_message = response
        if ack_message:
            final_message = ack_message

        # We don't append to history here because the frontend will do it
        logger.debug(f"Query processed successfully, is_report: {is_report}, is_visualization: {is_visualization}, response: {final_message}")

        # Prepare final payload fields
        report_html = ""
        report_markdown = ""

        if is_report and tool_response == "generate_detailed_report":
            # tool response is expected to be a JSON string with markdown & html
            try:
                parsed = json.loads(response)
                report_markdown = parsed.get("markdown", "")
                report_html = parsed.get("html", "")
            except Exception:
                # Return ONLY markdown with bold emphasis.
                # Wrap basic markdown in a simple container for backward compatibility
                report_html = f"<div class='prose prose-sm'>{response}</div>"
        else:
            # not a detailed report -> keep report fields empty
            report_markdown = ""
            report_html = ""

        # Increment user tokens if authenticated
        if current_user:
            increment_user_used_tokens(
                current_user,
                total_usage["total_tokens"],
                usage_record=usage_record,
            )

        return {
            "response": final_message,           # AI's short ack (for ChatPanel)
            "report_content": report_html,       # HTML to render in DisplayPanel
            "report_markdown": report_markdown,  # (optional) keep raw markdown if needed
            "summary": "",
            "anomalies": "",
            "insights": "",
            "is_report": is_report,
            "visualization": visualization,
            "total_input_tokens": total_usage["input_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"]
        }

    except Exception as e:
        logger.error(f"Error executing tool {tool_response}: {str(e)}")
        response = f"Error: Failed to execute tool '{tool_response}'. {str(e)}"
        
        # Still increment tokens even on error if authenticated
        if current_user:
            increment_user_used_tokens(
                current_user,
                total_usage["total_tokens"],
                usage_record=usage_record,
            )
        
        return {
            "response": response,
            "summary": "",
            "anomalies": "",
            "insights": "",
            "is_report": False,
            "visualization": None,
            "total_input_tokens": total_usage["input_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_tokens": total_usage["total_tokens"]
        }


async def create_or_update_db_from_data(data: str, filename: str, current_user: dict = None):
    global current_db_path
    
    usage_record = None
    try:
        # Ensure usage quota is available if user is authenticated
        if current_user:
            usage_record = ensure_usage_quota_available(current_user)

        logger.info(f"Creating/updating database for: {filename}")
        data_list = json.loads(data)
        
        if not isinstance(data_list, list) or len(data_list) < 1 or not isinstance(data_list[0], list):
            raise ValueError("Invalid data format: Expected a list of lists (JSON array of arrays).")
        
        headers = data_list[0]
        rows = data_list[1:] if len(data_list) > 1 else []
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert Excel serial dates to readable date strings
        logger.info("Checking for and converting Excel serial dates...")
        df = convert_excel_dates_in_dataframe(df)

        # Clean up any old database files before creating the new one
        for f in os.listdir(DB_DIR):
            if f.endswith('.db'):
                try:
                    os.remove(os.path.join(DB_DIR, f))
                except OSError as e:
                    logger.warning(f"Error removing old db file {f}: {e}")

        # Define the path for the new database
        db_filename = f"{os.path.splitext(filename)[0]}.db"
        db_path = os.path.join(DB_DIR, db_filename)
        
        # Use our reliable function from tools.py
        init_db_from_dataframe(df, db_path)
        current_db_path = db_path
        
        logger.info(f"Database created/updated successfully at: {db_path}")
        return {"message": "Data processed successfully."}

    except Exception as e:
        logger.error(f"Error in create_or_update_db_from_data: {str(e)}", exc_info=True)
        current_db_path = None # Reset path if creation failed
        return {"message": f"Error processing data: {str(e)}"}

@router.post("/query")
async def copilot_query(
    query: str = Form(...),
    conversation: List[str] = Form(default_factory=list),
    current_user: dict = Depends(get_current_user)
):
    """Handles user queries against the store operations database using the orchestrator."""
    return await orchestrator.process(query, conversation)

@router.post("/sync-now")
async def sync_now(current_user: dict = Depends(get_current_user)):
    """Manually trigger a data sync from MongoDB."""
    success = await sync_mongodb_to_sqlite(DB_PATH)
    if success:
        return {"message": "Data synchronized successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to sync data from MongoDB.")
