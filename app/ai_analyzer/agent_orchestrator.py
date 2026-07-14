import os
import sqlite3
import pandas as pd
import json
import logging
from dotenv import load_dotenv
from typing import Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from app.ai_analyzer.rag_system import RAGSystem

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "store_ops.db")

DB_SCHEMA_REFERENCE = """
Available SQLite Tables and Columns:
1. Table 'products':
   - id (TEXT)
   - product_code (TEXT, e.g. 'BP-PROD-001') - unique product code (matches inventory.code and sales.code)
   - product_name (TEXT, e.g. 'Castrol GTX 5W30 (1 Quart)')
   - category (TEXT, e.g. 'Automotive', 'Food', 'Beverages', 'Fuel')
   - brand (TEXT)
   - unit_price (REAL, selling price)
   - vendor_id (TEXT, matches vendors.id)
   - status (TEXT)

2. Table 'inventory':
   - id (TEXT)
   - store_id (TEXT, matches stores.id, e.g. 'BP-CHI-1024')
   - code (TEXT, matches products.product_code) - unique product SKU code
   - name (TEXT, e.g. 'Castrol GTX 5W30 (1 Quart)')
   - category (TEXT)
   - uom (TEXT, e.g. 'Litres', 'Units')
   - unit_price (REAL)
   - storage_location (TEXT)
   - rol (INTEGER, Reorder Level)
   - roq (INTEGER, Reorder Quantity)
   - colour (TEXT)
   - viscosity (TEXT)
   - vendor_id (TEXT, matches vendors.id)
   - lead_time_days (INTEGER)
   - current_stock (INTEGER, current inventory level)
   - safety_stock_level (INTEGER)
   - predicted_stockout_date (TEXT, date)
   - avg_daily_consumption (REAL)
   - recommended_roq (INTEGER)
   - order_by_date (TEXT, date)
   - pr_mr_status (TEXT)
   - risk_level (TEXT, 'Low', 'Medium', 'High')
   - service_level (INTEGER)

3. Table 'purchase_orders':
   - id (TEXT, e.g. 'PO-10000') - purchase order ID
   - vendor_id (TEXT, matches vendors.id)
   - vendor_name (TEXT)
   - store_id (TEXT, matches stores.id)
   - items (TEXT, JSON array of objects like [{"name": "Pringles...", "quantity": 100}])
   - total_amount (REAL)
   - status (TEXT, 'Delayed', 'Pending', 'Delivered')
   - order_date (TEXT, date)
   - expected_delivery_date (TEXT, date)
   - actual_delivery_date (TEXT, date or null)
   - expected_items (INTEGER)
   - received_items (INTEGER)
   - rejected_items (INTEGER)
   - accepted_items (INTEGER)
   - lead_time_days (REAL)
   - on_time_target (REAL)

4. Table 'vendors':
   - id (TEXT, e.g. 'VND-001') - vendor ID
   - name (TEXT)
   - contact (TEXT)
   - email (TEXT)
   - is_external (INTEGER, boolean 1/0)
   - type (TEXT)
   - region (TEXT)
   - city (TEXT)
   - onboarding_date (TEXT)
   - managed_by (TEXT)
   - contract_status (TEXT)
   - payment_terms (TEXT)

5. Table 'vendor_issues':
   - id (TEXT, e.g. 'ISSUE-1000')
   - vendor_id (TEXT)
   - vendor_name (TEXT)
   - issue_type (TEXT)
   - description (TEXT)
   - date_reported (TEXT, date)
   - status (TEXT, 'Open', 'Resolved')
   - priority (TEXT, 'High', 'Low')
   - related_po (TEXT, matches purchase_orders.id)

6. Table 'stores':
   - id (TEXT, e.g. 'BP-CHI-1024')
   - name (TEXT)
   - city (TEXT)
   - address (TEXT)
   - contact (TEXT)
   - active_since (TEXT)
   - products (TEXT, JSON list of products)

7. Table 'recommendations':
   - id (TEXT, e.g. 'REC-1')
   - type (TEXT)
   - description (TEXT)
   - action_label (TEXT)
   - priority (TEXT)

8. Table 'sales':
   - transaction_id (TEXT, e.g. 'TX-100001')
   - timestamp (TEXT, datetime format 'YYYY-MM-DD HH:MM:SS')
   - store_id (TEXT)
   - code (TEXT, matches products.product_code and inventory.code)
   - name (TEXT)
   - category (TEXT)
   - quantity (REAL or INTEGER)
   - price (REAL)
   - total_amount (REAL)
   - payment_method (TEXT)
"""

class AgentOrchestrator:
    MODEL_NAME = "mistral-medium-2505"
    FALLBACK_MODEL = "gemini-2.5-flash"

    def __init__(self):
        self.rag = RAGSystem()
        self.active_model = self.MODEL_NAME
        try:
            from app.services.llm import get_mistral_llm
            self.llm = get_mistral_llm()
            logger.info("[INIT] Agent Orchestrator initialized.")
            logger.info(f"[MODEL] Active LLM: {self.MODEL_NAME} via Mistral API")
        except Exception as e:
            logger.warning(f"[MODEL] Failed to load Mistral: {e}. Falling back to {self.FALLBACK_MODEL}")
            self.active_model = self.FALLBACK_MODEL
            api_key = os.getenv("GOOGLE_API_KEY")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.1
            )

    async def parse_intent(self, query: str, history_text: str) -> dict:
        """Determines the routing category and planning steps."""
        logger.info(f"[INTENT AGENT] Parsing user query: '{query[:80]}...' " if len(query) > 80 else f"[INTENT AGENT] Parsing user query: '{query}'")
        prompt = f"""
        You are the Intent and Planning Agent for a BP Store Manager AI Copilot.
        Analyze the user's operational query and categorize it into one of these categories:
        - "db": Questions needing database stats, inventory, POs, vendors, sales, or product analysis.
        - "rag": Questions about operating guidelines, SOPs, safety response plans, compliance, HR, cash reconciliations.
        - "live": Questions about active IoT sensor readings (temperature check, leakage check).
        - "external": Questions that need competitor pricing, local/public events near the store, national market trends (inflation, oil prices, FRED), current weather for cities/regions, or any general web search queries.
        - "workflow": Action requests (e.g. reordering items, updates, creating escalation tickets).
        - "conversational": Greetings, small talk, general questions unrelated to store operations.

        Available Database Schema context for deciding "db" queries:
        {DB_SCHEMA_REFERENCE}

        Provide the output in JSON format:
        {{
          "category": "db" | "rag" | "live" | "external" | "workflow" | "conversational",
          "plan": "One sentence describing how you will resolve this request.",
          "is_complex": true | false
        }}

        Current Query: "{query}"
        Conversation History:
        {history_text}
        
        Return ONLY the raw JSON block.
        """
        try:
            res = await self.llm.ainvoke(prompt)
            content = res.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            logger.info(f"[INTENT AGENT] Routed to category: '{parsed.get('category')}' | Plan: {parsed.get('plan','')}")
            return parsed
        except Exception as e:
            logger.warning(f"[INTENT AGENT] Error parsing intent: {e}. Defaulting to 'db'.")
            return {"category": "db", "plan": "Default query resolution path.", "is_complex": False}

    def execute_db_query(self, sql: str) -> Tuple[list, list, str]:
        """Runs the SQL query on SQLite and returns headers, rows, and status message."""
        cleaned_sql = self._clean_sql_query(sql)
        logger.info(f"[DB AGENT] Executing SQL:\n{cleaned_sql}")
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(cleaned_sql)
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            logger.info(f"[DB AGENT] Query returned {len(rows)} row(s) with headers: {headers}")
            return headers, rows, "success"
        except Exception as e:
            logger.error(f"[DB AGENT] SQL execution error: {e}")
            return [], [], str(e)

    def _clean_sql_query(self, sql_str: str) -> str:
        """Cleans and extracts the raw SQL statement from LLM output, removing any conversational wrappers."""
        cleaned = sql_str.strip()
        # 1. Check for standard markdown code blocks
        if "```sql" in cleaned:
            cleaned = cleaned.split("```sql")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        else:
            # 2. Extract starting from SQL keywords if conversational wrapper prefix is present
            keywords = ["SELECT", "WITH", "UPDATE", "INSERT", "DELETE"]
            for kw in keywords:
                idx = cleaned.upper().find(kw)
                if idx != -1:
                    cleaned = cleaned[idx:].strip()
                    break
        # 3. Strip any trailing comments or markdown code fences
        if "```" in cleaned:
            cleaned = cleaned.split("```")[0].strip()
        # 4. Truncate after the first semicolon if there's conversational suffix text
        if ";" in cleaned:
            parts = cleaned.split(";")
            cleaned = parts[0].strip() + ";"
        return cleaned

    async def generate_sql(self, query: str, history_text: str, role: str = None, store_id: str = None, region: str = None) -> str:
        """Generates a valid SQLite SQL query from natural language."""
        logger.info(f"[SQL AGENT] Generating SQL for query: '{query[:80]}...' " if len(query) > 80 else f"[SQL AGENT] Generating SQL for query: '{query}'")
        logger.info(f"[SQL AGENT] User context: role={role}, storeId={store_id}, region={region}")
        
        scope_instructions = ""
        if role == "store manager" and store_id:
            scope_instructions = f"""
            - SECURITY ENFORCEMENT: The current user has the role of a STORE MANAGER assigned ONLY to Store ID '{store_id}'.
            - You MUST strictly restrict ALL SQL queries to select data where `store_id = '{store_id}'`.
            - Apply this store filter to any tables containing store-specific data, such as `inventory` (match `store_id = '{store_id}'`) and `purchase_orders` (match `store_id = '{store_id}'`).
            - Do not allow the user to see data or aggregate metrics for other stores under any circumstances.
            """
        elif role == "vendor manager" and region:
            scope_instructions = f"""
            - SECURITY ENFORCEMENT: The current user has the role of a VENDOR MANAGER assigned ONLY to Region '{region}'.
            - You MUST strictly restrict ALL SQL queries to select data within the region '{region}'.
            - If querying the `vendors` or `vendor_issues` tables, match `region = '{region}'`.
            - If querying tables like `purchase_orders` or `inventory` which do not have a direct `region` column, you MUST JOIN with `vendors` or `stores` on `vendor_id` or `store_id` to enforce that the vendor's or store's region is '{region}'.
            - Do not allow the user to see data or aggregate metrics for other regions under any circumstances.
            """
        else:
            scope_instructions = """
            - The user has unrestricted SUPER ADMIN access. No scope filters are needed.
            """

        rules_block = """
        Rules:
        - Return ONLY the clean SQLite query. No markdown wrapper (do NOT wrap in ```sql).
        - Use clean JOINs if matching products with inventory or POs.
        - Handle dates: use `date('now')` for the current date instead of CURRENT_DATE.
        - Case insensitive checks: use `LIKE` for text filters.
        - The column `avg_daily_consumption` and safety stock fields only exist in the `inventory` table, NOT the `products` table.
        - Avoid integer division: always multiply by 1.0 before dividing (e.g. `received_items * 1.0 / expected_items`).
        - To compute Vendor Performance Metrics safely, use SUM(CASE WHEN ...) for conditional counts inside aggregates.
        - ORDERING: For all SQL queries, always order the output in ASCENDING order based on primary columns (e.g. classification names, product codes, totals, or item counts) using ORDER BY ... ASC, unless specified otherwise by the user.
        """

        prompt = f"""
        You are the SQLite Database Agent for a BP Store.
        Convert the user's natural language request into a single valid SQLite SQL query.
        
        {DB_SCHEMA_REFERENCE}

        {scope_instructions}

        FORMULAS AND BUSINESS RULES FOR METRICS AND FIELDS:
        1. Inventory & Stock Metrics:
           - Total Products: Count of Unique SKU Codes in the Store Catalog (`COUNT(DISTINCT code)` in products/inventory).
           - Inventory Value: `SUM(current_stock * unit_price)` in inventory.
           - Critical Stockout: Count of products where Days Left <= 7 (where Days Left = `FLOOR(current_stock / avg_daily_consumption)`).
           - Below Reorder (PR Needed): Count of products where `current_stock <= rol` (Reorder Level).
           - Overdue Orders: Count of purchase orders where `expected_delivery_date < date('now')` AND `status NOT IN ('Delivered', 'Cancelled')` (orders past due and not yet delivered).
           - Safety Stock Level: `CEIL(avg_daily_consumption * 0.5 * lead_time_days)`
           - Days Left: `FLOOR(current_stock / avg_daily_consumption)`
           - Predicted Stockout Date: `date('now', '+' || (current_stock / avg_daily_consumption) || ' days')`
           - Recommended ROQ (Reorder Quantity): `ROUND(avg_daily_consumption * 15)`
           - Order By Date: `date('now', '+' || (current_stock / avg_daily_consumption - lead_time_days) || ' days')`
           - Status (Monitor / PR / MR):
             * If `current_stock > rol` -> 'Monitor'
             * Else If `FLOOR(current_stock / avg_daily_consumption) >= lead_time_days` -> 'PR (Purchase Request)'
             * Else -> 'MR (Material Required)'

        2. FSN (Fast, Slow, Non-Moving) Analysis Classification:
           - Fast Moving (F): `avg_daily_consumption >= 25`
           - Slow Moving (S): `5 <= avg_daily_consumption < 25`
           - Non-Moving (N): `avg_daily_consumption < 5`

        3. Vendor Performance Score:
           - Fill Rate: `(received_items * 100.0) / expected_items` (on delivered or all POs)
           - On-Time Delivery Rate: `(Count of delivered POs on-time * 100.0) / Total delivered POs`. (An order is on-time if `actual_delivery_date <= expected_delivery_date` or `status != 'Delayed'`).
           - Rejection Rate: `(rejected_items * 100.0) / received_items` (where `received_items` > 0).
           - Order Accuracy: `100.0 - Rejection Rate`
           - Stockout Score: `MAX(0, 100 - (Count of delayed POs for vendor * 5))`
           - Lead Time Score: `MAX(0, 100 - (AVG(actual_lead_time_days) * 10))`
           - Vendor Score: `0.40 * Fill Rate + 0.25 * On-Time Delivery Rate + 0.20 * Order Accuracy + 0.10 * Stockout Score + 0.05 * Lead Time Score`
           - Organisation Weighted Score: `Vendor Score * (accepted_items * 1.0 / received_items)` where `accepted_items = received_items - rejected_items`. (If `received_items` = 0, use `Vendor Score`).

        4. Vendor Status Criteria:
           - Excellent: Weighted Score >= 95
           - Good: 85 <= Weighted Score < 95
           - Needs Improvement: 75 <= Weighted Score < 85
           - Critical: Weighted Score < 75

        5. Vendor Action Center Rules (Issue Generation):
           - Rejection Rate >= 5% -> "Excessive Quality Rejection (High Priority)"
           - Fill Rate < 90% -> "SLA Fill Rate Breach (High Priority)"
           - Days Late >= 5 -> "Critical Delivery Delay (High Priority)"
           - Days Late = 4 -> "Shipment Delayed (Medium Priority)"
           - Days Late = 3 -> "Invoice / Manifest Mismatch (Medium Priority)"
           - Days Late < 3 -> "Slight Delivery Delay (Low Priority)"

        {rules_block}
        
        User Query: "{query}"
        History context:
        {history_text}
        """
        res = await self.llm.ainvoke(prompt)
        sql = res.content.strip()
        if sql.startswith("```sql"):
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif sql.startswith("```"):
            sql = sql.split("```")[1].split("```")[0].strip()
        logger.info(f"[SQL AGENT] Generated SQL:\n{sql}")
        return sql.strip(";")

    def get_live_sensor_readings(self) -> dict:
        """Fetches simulated real-time IoT metrics."""
        return {
            "sensors": [
                {"sensor_id": "SEN-001", "name": "Wild Bean Milk Chiller", "type": "Temperature", "value": "3.4 °C", "status": "Normal"},
                {"sensor_id": "SEN-002", "name": "Forecourt Leakage Detector", "type": "Leakage", "value": "0.0 ppm", "status": "Normal"},
                {"sensor_id": "SEN-003", "name": "Underground Tank 1 Level", "type": "Level", "value": "12,450 L", "status": "Normal"},
                {"sensor_id": "SEN-004", "name": "Bakery Sandwich Freezer", "type": "Temperature", "value": "-18.2 °C", "status": "Normal"}
            ],
            "weather": {
                "location": "BP Loop Connect, Chicago IL",
                "condition": "Light rain",
                "temperature": "14.5 °C",
                "wind": "18.2 km/h East"
            }
        }

    async def process(self, query: str, conversation: list, current_user: dict = None) -> dict:
        """Main orchestrator entrypoint."""
        # Convert history format
        history_lines = []
        for c in conversation:
            try:
                msg = json.loads(c) if isinstance(c, str) else c
                history_lines.append(f"User: {msg.get('user', '')}\nAI: {msg.get('ai', '')}")
            except Exception:
                pass
        history_text = "\n".join(history_lines)

        intent = await self.parse_intent(query, history_text)
        category = intent.get("category", "conversational")
        
        db_results = None
        rag_results = None
        live_results = None
        workflow_executed = None
        sql_generated = None
        
        # 1. DB Agent
        if category == "db":
            role = current_user.get("role") if current_user else None
            store_id = current_user.get("storeId") if current_user else None
            region = current_user.get("region") if current_user else None
            logger.info(f"[DB AGENT] Starting. User role={role}, storeId={store_id}, region={region}")
            sql = await self.generate_sql(query, history_text, role, store_id, region)
            sql_generated = sql
            headers, rows, status = self.execute_db_query(sql)
            if status == "success":
                db_results = {
                    "headers": headers,
                    "rows": [list(row) for row in rows]
                }
                logger.info(f"[DB AGENT] Complete. Returned {len(rows)} row(s).")
            else:
                db_results = {"error": status}
                logger.error(f"[DB AGENT] Query failed: {status}")

        # 2. RAG Agent
        elif category == "rag":
            logger.info("[RAG AGENT] Starting semantic policy/SOP search.")
            rag_docs = self.rag.search(query, top_k=2)
            rag_results = [
                {"title": r["doc"]["title"], "category": r["doc"]["category"], "content": r["doc"]["content"], "score": r["score"]}
                for r in rag_docs
            ]
            logger.info(f"[RAG AGENT] Retrieved {len(rag_results)} document(s).")

        # 3. Live IoT Agent
        elif category == "live":
            logger.info("[LIVE AGENT] Fetching real-time IoT sensor readings.")
            live_results = self.get_live_sensor_readings()
            logger.info(f"[LIVE AGENT] Retrieved {len(live_results.get('sensors', []))} sensor reading(s).")

        # 4. External Web Services Agent
        elif category == "external":
            logger.info("[EXTERNAL AGENT] Starting classification of external query.")
            
            # Resolve user's city if storeId is provided
            user_city = None
            store_id = current_user.get("storeId") if current_user else None
            if store_id:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT city FROM stores WHERE id = ?", (store_id,))
                    row = cursor.fetchone()
                    if row:
                        user_city = row[0]
                    conn.close()
                    logger.info(f"[EXTERNAL AGENT] Resolved store city: '{user_city}' for store ID '{store_id}'")
                except Exception as e:
                    logger.warning(f"[EXTERNAL AGENT] Failed to fetch city for store: {e}")

            classification_prompt = f"""
            Classify the user's query into one of these specific external tools:
            - "search": General web searches, news lookup, or generic questions.
            - "weather": Weather conditions, forecasts.
            - "pricing": Competitor pricing, convenience product/gas competitor price check.
            - "events": Concerts, festivals, sports, local events.
            - "trends": National economic market trends (inflation, CPI, FRED, unemployment).

            Provide output in JSON format:
            {{
                "tool": "search" | "weather" | "pricing" | "events" | "trends",
                "param": "The key parameter (e.g. search query, city name, product name, or economic metric type like 'inflation')"
            }}

            Context Context Guidelines:
            - User's Assigned Store ID: {store_id if store_id else 'None'}
            - Store City: {user_city if user_city else 'None'}
            - User's Assigned Region: {current_user.get('region') if current_user else 'None'}

            If the user asks relative questions (e.g., "my store", "local weather", "events nearby", "gas price comparison"), use the Store City ("{user_city}") or context details to fill the parameter.

            User Query: "{query}"
            Return ONLY the raw JSON block.
            """
            try:
                class_res = await self.llm.ainvoke(classification_prompt)
                class_content = class_res.content.strip()
                if class_content.startswith("```json"):
                    class_content = class_content.split("```json")[1].split("```")[0].strip()
                elif class_content.startswith("```"):
                    class_content = class_content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(class_content)
                tool = parsed.get("tool", "search")
                param = parsed.get("param", query)
                
                logger.info(f"[EXTERNAL AGENT] Selected tool: '{tool}' | Parameter: '{param}'")
                
                if tool == "search":
                    from app.external_services.web_search import search_web
                    external_results = await search_web(param)
                elif tool == "weather":
                    from app.external_services.weather_service import get_weather
                    external_results = await get_weather(param)
                elif tool == "pricing":
                    from app.external_services.competitor_pricing import get_competitor_pricing
                    external_results = await get_competitor_pricing(param)
                elif tool == "events":
                    from app.external_services.event_tracker import get_local_events
                    external_results = await get_local_events(param)
                elif tool == "trends":
                    from app.external_services.market_trends import get_market_trend
                    external_results = await get_market_trend(param)
                else:
                    external_results = "Unknown tool requested."
            except Exception as e:
                logger.warning(f"[EXTERNAL AGENT] Classification failed: {e}. Defaulting to Tavily search.")
                from app.external_services.web_search import search_web
                external_results = await search_web(query)

        # 5. Workflow Agent
        elif category == "workflow":
            logger.info("[WORKFLOW AGENT] Executing workflow action.")
            if "reorder" in query.lower() or "order" in query.lower():
                workflow_executed = "Draft Purchase Order created. Assigned ID 'PO-99120'. Items matched to reorder quantities (ROQ). Final manager signature required in ERP/SAP."
            else:
                workflow_executed = "Escalation ticket generated: 'Vendor Delivery Issue VND-001' logged in operations log under ID 'TKT-82190'."
            logger.info(f"[WORKFLOW AGENT] Result: {workflow_executed}")

        # Synthesis Agent
        logger.info(f"[SYNTHESIS AGENT] Composing final response using model: {self.active_model}")
        synthesis_prompt = f"""
        You are the BP Store Manager AI Copilot.
        Respond to the user's store operations query based on the fetched context and execution details.
        
        User Query: "{query}"
        Category Routed: {category}
        
        Context Details:
        - DB Execution Results: {json.dumps(db_results) if db_results else "None"}
        - RAG Retrieval: {json.dumps(rag_results) if rag_results else "None"}
        - Live IoT Sensors: {json.dumps(live_results) if live_results else "None"}
        - Workflow Executed: {workflow_executed if workflow_executed else "None"}
        - External Web Context: {external_results if (category == 'external' and 'external_results' in locals() and external_results) else "None"}
        
        Response Guidelines:
        - Speak like an expert Store Manager Copilot. Maintain utmost professionalism and keep answers concise.
        - CRITICAL: Avoid long, dense paragraph blocks. Never write more than 2 consecutive sentences of paragraph text.
        - CRITICAL: Present lists, parameters, comparisons, or details using clean **Markdown Tables**, **bullet points**, or **bold structured lists** so it is easy to read at a glance.
        - CRITICAL CONFLICT RESOLUTION: If both DB Execution Results and RAG Retrieval or Conversation History are present, you MUST prioritize the numbers, counts, quantities, and statistics in the 'DB Execution Results'. Do NOT copy or use any counts, values, or statistics mentioned in RAG guidelines, SOPs, or conversation history (e.g. if the current DB results show Fast: 29, you MUST output 29, even if history or SOP documents mention different counts like 19). Double-check that your written numbers match the current execution values exactly, not the history.
        - CRITICAL: You must extract and copy the numbers and stats from the DB Execution Results exactly. Do NOT guess, alter, or count them incorrectly (e.g., FSN counts, stock levels, or prices must match the DB results 1:1). If the DB results show 19 items, you MUST output 19.
        - IMPORTANT: Output your final response in clean Markdown. Do NOT wrap the entire response in code fences.
        - TABLE RULE:
          * If a table is needed and not loaded in the right panel, you MUST render it as a standard **Markdown Table** inside this text response.
          * Even if data is shown in the right panel, present the key summaries in the chat as structured bullet points or short tables.
        - If sensor temp warnings exceed standard limits, flag it as a Warning.
        - CRITICAL: Do NOT include, show, or reference the internal SQL query. Never show SQL code. Present only insights in business language.
        """
        
        res = await self.llm.ainvoke(synthesis_prompt)
        final_answer = res.content.strip()
        
        # Extract token usage from response metadata
        token_usage = {}
        try:
            usage = res.response_metadata.get("token_usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            logger.info(f"[SYNTHESIS AGENT] Done. Tokens used — prompt: {token_usage['prompt_tokens']}, completion: {token_usage['completion_tokens']}, total: {token_usage['total_tokens']}")
        except Exception:
            pass
        
        # Structure payload matching display panels
        visualization = None
        table_data = None
        report_markdown = None
        
        if db_results and "headers" in db_results and len(db_results["rows"]) > 0:
            table_data = [db_results["headers"]] + db_results["rows"]
            visualization = self._auto_visualize(query, db_results)
            if visualization:
                logger.info(f"[VIZ AGENT] Auto-generated chart: type={visualization.get('type')}, title={visualization.get('title')}")
        elif rag_results:
            report_markdown = final_answer
        elif category == "external" and 'external_results' in locals() and external_results:
            # Show raw external details in report panel, summary in chat
            report_markdown = f"# External Query Source Details\n\nQuery: {query}\n\n{external_results}"
            
        return {
            "response": final_answer,
            "tableData": table_data,
            "chartData": visualization,
            "reportData": report_markdown,
            "is_report": report_markdown is not None,
            "visualization": visualization,
            "model_name": self.active_model,
            "token_usage": token_usage
        }

    def _auto_visualize(self, query: str, db_results: dict) -> dict | None:
        """Intelligently auto-generates a Chart.js JSON config from query intent and DB results."""
        q = query.lower()
        headers = db_results.get("headers", [])
        rows = db_results.get("rows", [])
        if not headers or not rows:
            return None
        h_lower = [h.lower() for h in headers]

        # --- FSN Analysis -> Donut Chart ---
        if any(kw in q for kw in ["fsn", "fast moving", "slow moving", "non moving", "non-moving"]):
            fsn_col = next((i for i, h in enumerate(h_lower) if h in ["fsn_classification", "classification", "fsn", "fsn classification"]), None)
            count_col = next((i for i, h in enumerate(h_lower) if "count" in h or h == "total"), None)
            if fsn_col is not None:
                fsn_counts = {"F": 0, "S": 0, "N": 0}
                for row in rows:
                    label = str(row[fsn_col]).strip().upper()
                    count = int(row[count_col]) if count_col is not None else 1
                    if "FAST" in label or label.startswith("F"):
                        fsn_counts["F"] += count
                    elif "SLOW" in label or label.startswith("S"):
                        fsn_counts["S"] += count
                    elif "NON" in label or label.startswith("N"):
                        fsn_counts["N"] += count
                if sum(fsn_counts.values()) > 0:
                    return {
                        "type": "doughnut",
                        "title": "FSN Classification Breakdown",
                        "labels": ["Fast Moving (F)", "Slow Moving (S)", "Non-Moving (N)"],
                        "datasets": [{
                            "label": "SKU Count",
                            "data": [fsn_counts["F"], fsn_counts["S"], fsn_counts["N"]],
                            "backgroundColor": ["#10b981", "#f59e0b", "#ef4444"]
                        }]
                    }

        # --- Risk / Stockout levels -> Donut ---
        if any(kw in q for kw in ["risk", "stockout risk", "risk level", "risk summary"]):
            risk_col = next((i for i, h in enumerate(h_lower) if "risk" in h or "level" in h), None)
            count_col = next((i for i, h in enumerate(h_lower) if "count" in h or "total" in h or "products" in h), None)
            if risk_col is not None and count_col is not None and len(rows) > 1:
                color_map = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981", "healthy": "#10b981"}
                labels = [str(r[risk_col]) for r in rows]
                data = [float(r[count_col]) if r[count_col] is not None else 0 for r in rows]
                colors = [color_map.get(l.lower(), "#6366f1") for l in labels]
                return {
                    "type": "doughnut",
                    "title": "Stockout Risk Distribution",
                    "labels": labels,
                    "datasets": [{"label": "Products", "data": data, "backgroundColor": colors}]
                }

        # --- Category breakdown -> Bar ---
        if any(kw in q for kw in ["category", "by category", "per category", "categories"]):
            cat_col = next((i for i, h in enumerate(h_lower) if "category" in h), None)
            val_col = next((i for i, h in enumerate(h_lower) if h not in ["category"] and
                any(kw in h for kw in ["count", "total", "value", "stock", "amount"])), None)
            if val_col is None and len(headers) > 1:
                val_col = 1
            if cat_col is not None and val_col is not None:
                labels = [str(r[cat_col]) for r in rows[:12]]
                data = []
                for r in rows[:12]:
                    try: data.append(float(r[val_col]))
                    except: data.append(0)
                return {
                    "type": "bar",
                    "title": f"{headers[val_col].replace('_',' ').title()} by Category",
                    "labels": labels,
                    "datasets": [{"label": headers[val_col].replace('_',' ').title(), "data": data, "backgroundColor": "#6366f1"}]
                }

        # --- Vendor performance -> Bar ---
        if any(kw in q for kw in ["vendor", "supplier", "vendor score", "performance"]):
            name_col = next((i for i, h in enumerate(h_lower) if "name" in h or "vendor" in h), None)
            score_col = next((i for i, h in enumerate(h_lower) if "score" in h or "rate" in h or "fill" in h), None)
            if name_col is not None and score_col is not None:
                labels = [str(r[name_col]) for r in rows[:10]]
                data = []
                for r in rows[:10]:
                    try: data.append(round(float(r[score_col]), 2))
                    except: data.append(0)
                return {
                    "type": "bar",
                    "title": "Vendor Performance Score",
                    "labels": labels,
                    "datasets": [{"label": headers[score_col].replace('_',' ').title(), "data": data, "backgroundColor": "#3b82f6"}]
                }

        # --- Spend / amount -> Bar ---
        if any(kw in q for kw in ["spend", "amount", "cost", "purchase"]):
            val_col = next((i for i, h in enumerate(h_lower) if
                any(kw in h for kw in ["amount", "spend", "cost", "total"])), None)
            if val_col is None and len(headers) > 1:
                val_col = 1
            if val_col is not None:
                labels = [str(r[0]) for r in rows[:10]]
                data = []
                for r in rows[:10]:
                    try: data.append(round(float(r[val_col]), 2))
                    except: data.append(0)
                return {
                    "type": "bar",
                    "title": headers[val_col].replace('_',' ').title(),
                    "labels": labels,
                    "datasets": [{"label": headers[val_col].replace('_',' ').title(), "data": data, "backgroundColor": "#8b5cf6"}]
                }

        # --- Generic 2-column numeric result -> Bar ---
        if len(headers) >= 2:
            try:
                data = [float(r[1]) for r in rows[:12]]
                if any(d != 0 for d in data):
                    return {
                        "type": "bar",
                        "title": headers[1].replace('_',' ').title(),
                        "labels": [str(r[0]) for r in rows[:12]],
                        "datasets": [{"label": headers[1].replace('_',' ').title(), "data": data, "backgroundColor": "#0ea5e9"}]
                    }
            except Exception:
                pass

        return None

    async def process_stream(self, query: str, conversation: list, current_user: dict = None):
        """Asynchronous generator yielding SSE chunks for progress steps, tokens, and final result."""
        # 1. Parse history
        history_lines = []
        for c in conversation:
            try:
                msg = json.loads(c) if isinstance(c, str) else c
                history_lines.append(f"User: {msg.get('user', '')}\nAI: {msg.get('ai', '')}")
            except Exception:
                pass
        history_text = "\n".join(history_lines)

        # Yield initial intent parsing step
        yield json.dumps({"type": "step", "message": "Analyzing query intent..."})

        intent = await self.parse_intent(query, history_text)
        category = intent.get("category", "conversational")
        
        db_results = None
        rag_results = None
        live_results = None
        workflow_executed = None
        sql_generated = None
        external_results = None
        tools_used = []

        # Yield routed step
        yield json.dumps({"type": "step", "message": f"Query routed to category: '{category}'"})

        # DB Agent
        if category == "db":
            yield json.dumps({"type": "step", "message": "Formulating database query..."})
            role = current_user.get("role") if current_user else None
            store_id = current_user.get("storeId") if current_user else None
            region = current_user.get("region") if current_user else None
            
            sql = await self.generate_sql(query, history_text, role, store_id, region)
            sql_generated = sql
            
            yield json.dumps({"type": "step", "message": "Querying local store database..."})
            headers, rows, status = self.execute_db_query(sql)
            if status == "success":
                db_results = {
                    "headers": headers,
                    "rows": [list(row) for row in rows]
                }
                tools_used.append("SQLite Database")
            else:
                db_results = {"error": status}

        # RAG Agent
        elif category == "rag":
            yield json.dumps({"type": "step", "message": "Searching policy SOP knowledge base..."})
            rag_docs = self.rag.search(query, top_k=2)
            rag_results = [
                {"title": r["doc"]["title"], "category": r["doc"]["category"], "content": r["doc"]["content"], "score": r["score"]}
                for r in rag_docs
            ]
            tools_used.append("SOP Guidelines (RAG)")

        # Live IoT Agent
        elif category == "live":
            yield json.dumps({"type": "step", "message": "Reading real-time IoT sensors..."})
            live_results = self.get_live_sensor_readings()
            tools_used.append("IoT Sensors Link")

        # External Agent
        elif category == "external":
            yield json.dumps({"type": "step", "message": "Classifying external lookup..."})
            
            user_city = None
            store_id = current_user.get("storeId") if current_user else None
            if store_id:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT city FROM stores WHERE id = ?", (store_id,))
                    row = cursor.fetchone()
                    if row: user_city = row[0]
                    conn.close()
                except Exception:
                    pass

            classification_prompt = f"""
            Classify the user's query into one of these specific external tools:
            - "search": General web searches, news lookup, or generic questions.
            - "weather": Weather conditions, forecasts.
            - "pricing": Competitor pricing, convenience product/gas competitor price check.
            - "events": Concerts, festivals, sports, local events.
            - "trends": National economic market trends (inflation, CPI, FRED, unemployment).

            Provide output in JSON format:
            {{
                "tool": "search" | "weather" | "pricing" | "events" | "trends",
                "param": "The key parameter (e.g. search query, city name, product name, or economic metric type like 'inflation')"
            }}

            Context Context Guidelines:
            - User's Assigned Store ID: {store_id if store_id else 'None'}
            - Store City: {user_city if user_city else 'None'}
            - User's Assigned Region: {current_user.get('region') if current_user else 'None'}

            If the user asks relative questions, use the Store City ("{user_city}") or context details to fill the parameter.

            User Query: "{query}"
            Return ONLY the raw JSON block.
            """
            try:
                class_res = await self.llm.ainvoke(classification_prompt)
                class_content = class_res.content.strip()
                if class_content.startswith("```json"):
                    class_content = class_content.split("```json")[1].split("```")[0].strip()
                elif class_content.startswith("```"):
                    class_content = class_content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(class_content)
                tool = parsed.get("tool", "search")
                param = parsed.get("param", query)
                
                if tool == "search":
                    yield json.dumps({"type": "step", "message": f"Searching web for: '{param}'..."})
                    from app.external_services.web_search import search_web
                    external_results = await search_web(param)
                    tools_used.append("Tavily Search")
                elif tool == "weather":
                    yield json.dumps({"type": "step", "message": f"Fetching weather forecast for: '{param}'..."})
                    from app.external_services.weather_service import get_weather
                    external_results = await get_weather(param)
                    tools_used.append("Weather API")
                elif tool == "pricing":
                    yield json.dumps({"type": "step", "message": f"Querying competitor pricing for: '{param}'..."})
                    from app.external_services.competitor_pricing import get_competitor_pricing
                    external_results = await get_competitor_pricing(param)
                    tools_used.append("Apify Pricing Scraper")
                elif tool == "events":
                    yield json.dumps({"type": "step", "message": f"Checking local events near: '{param}'..."})
                    from app.external_services.event_tracker import get_local_events
                    external_results = await get_local_events(param)
                    tools_used.append("PredictHQ Events")
                elif tool == "trends":
                    yield json.dumps({"type": "step", "message": f"Querying FRED market index for: '{param}'..."})
                    from app.external_services.market_trends import get_market_trend
                    external_results = await get_market_trend(param)
                    tools_used.append("FRED Economic Trends")
            except Exception:
                yield json.dumps({"type": "step", "message": "Searching web..."})
                from app.external_services.web_search import search_web
                external_results = await search_web(query)
                tools_used.append("Tavily Search")

        # Workflow Agent
        elif category == "workflow":
            yield json.dumps({"type": "step", "message": "Logging workflow transaction..."})
            if "reorder" in query.lower() or "order" in query.lower():
                workflow_executed = "Draft Purchase Order created. Assigned ID 'PO-99120'. Items matched to reorder quantities (ROQ). Final manager signature required in ERP/SAP."
            else:
                workflow_executed = "Escalation ticket generated: 'Vendor Delivery Issue VND-001' logged in operations log under ID 'TKT-82190'."
            tools_used.append("Workflow Automation")

        # Final Synthesis
        yield json.dumps({"type": "step", "message": "Synthesizing response..."})
        
        table_in_panel = True
        table_data = None
        
        if db_results and "headers" in db_results and len(db_results["rows"]) > 0:
            num_rows = len(db_results["rows"])
            num_cols = len(db_results["headers"])
            if num_rows <= 5 and num_cols <= 3:
                table_in_panel = False
                logger.info(f"[ORCHESTRATOR] Small table detected ({num_rows}x{num_cols}). Rendering directly in chat.")
            else:
                table_data = [db_results["headers"]] + db_results["rows"]

        synthesis_prompt = f"""
        You are the BP Store Manager AI Copilot.
        Respond to the user's store operations query based on the fetched context, history context, and execution details.
        
        User Query: "{query}"
        Category Routed: {category}
        
        Conversation History:
        {history_text}
        
        Context Details:
        - DB Execution Results: {json.dumps(db_results) if db_results else "None"}
        - RAG Retrieval: {json.dumps(rag_results) if rag_results else "None"}
        - Live IoT Sensors: {json.dumps(live_results) if live_results else "None"}
        - Workflow Executed: {workflow_executed if workflow_executed else "None"}
        - External Web Context: {external_results if external_results else "None"}
        
        Response Guidelines:
        - Speak like an expert Store Manager Copilot. Maintain utmost professionalism and keep answers concise.
        - CRITICAL: Avoid long, dense paragraph blocks. Never write more than 2 consecutive sentences of paragraph text.
        - CRITICAL: Present lists, parameters, comparisons, or details using clean **Markdown Tables**, **bullet points**, or **bold structured lists** so it is easy to read at a glance.
        - CRITICAL CONFLICT RESOLUTION: If both DB Execution Results and RAG Retrieval or Conversation History are present, you MUST prioritize the numbers, counts, quantities, and statistics in the 'DB Execution Results'. Do NOT copy or use any counts, values, or statistics mentioned in RAG guidelines, SOPs, or conversation history (e.g. if the current DB results show Fast: 29, you MUST output 29, even if history or SOP documents mention different counts like 19). Double-check that your written numbers match the current execution values exactly, not the history.
        - CRITICAL: You must extract and copy the numbers and stats from the DB Execution Results exactly. Do NOT guess, alter, or count them incorrectly (e.g., FSN counts, stock levels, or prices must match the DB results 1:1). If the DB results show 19 items, you MUST output 19.
        - IMPORTANT: Output your final response in clean Markdown. Do NOT wrap the entire response in code fences.
        - TABLE RULE:
          * If table_in_panel is FALSE, you MUST display the results clearly in a standard **Markdown Table** right here in your text response.
          * If table_in_panel is TRUE, do NOT render any markdown table in your text response (it will be loaded in the Analytics panel). However, you should still present a concise summary of key highlights using bullet points or mini tables here.
        - Value parameters: table_in_panel = {table_in_panel}
        - If sensor temp warnings exceed standard limits, flag it as a Warning.
        - CRITICAL: Do NOT include, show, or reference the internal SQL query. Never show SQL code. Present only insights in business language.
        """

        full_text = ""
        async for chunk in self.llm.astream(synthesis_prompt):
            content = chunk.content
            full_text += content
            yield json.dumps({"type": "token", "text": content})

        visualization = None
        report_markdown = None
        
        if table_data:
            visualization = self._auto_visualize(query, db_results)
        elif rag_results:
            report_markdown = full_text
        elif category == "external" and external_results:
            report_markdown = f"# External Query Source Details\n\nQuery: {query}\n\n{external_results}"

        yield json.dumps({
            "type": "result",
            "tableData": table_data,
            "chartData": visualization,
            "reportData": report_markdown,
            "tools_used": tools_used,
            "model_name": self.active_model
        })
