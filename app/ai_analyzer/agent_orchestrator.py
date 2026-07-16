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

9. Table 'purchase_history':
   - id (TEXT, unique mongo ID)
   - purchase_id (TEXT, e.g. 'PUR-20260714115928-8525')
   - store_id (TEXT, matches stores.id)
   - customer_id (TEXT, e.g. 'CUS-0021')
   - customer_name (TEXT, e.g. 'Noah Surname1')
   - items (TEXT, JSON array of purchased items, e.g. '[{"product_code": "BP-PROD-081", "product_name": "Milk", "category": "Grocery", "quantity": 1, "unit_price": 28.99, "total_price": 28.99}]')
   - total_amount (REAL, total purchase cost)
   - item_count (INTEGER, total items in purchase)
   - purchase_timestamp (TEXT/TIMESTAMP, purchase date/time, e.g. '2026-07-14 11:59:28')
   - payment_method (TEXT, e.g. 'Credit Card', 'Debit Card')
"""

class AgentOrchestrator:
    MODEL_NAME = "claude-haiku-4.5"
    FALLBACK_MODEL = "gemini-2.5-flash"

    def __init__(self):
        self.rag = RAGSystem()
        self.active_model = self.MODEL_NAME
        try:
            from app.services.llm import get_mistral_llm,get_openrouter_llm
            self.llm = get_openrouter_llm()
            logger.info("[INIT] Agent Orchestrator initialized.")
            logger.info(f"[MODEL] Active LLM: {self.MODEL_NAME} via Openrouter API")
        except Exception as e:
            logger.warning(f"[MODEL] Failed to load Openrouter: {e}. Falling back to {self.FALLBACK_MODEL}")
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
        Analyze the user's operational query and categorize it into one of these categories:
        - "db": Questions needing database stats, inventory, POs, vendors, sales, product analysis, store profile highlights, or KPI metrics summaries (e.g., total products count, inventory valuation, stockout counts, reorder status, or overdue purchase orders).
        - "rag": Questions about operating guidelines, SOPs, safety response plans, compliance, HR, cash reconciliations. Do NOT route store profile/performance highlights here.
        - "live": Questions about active IoT sensor readings (temperature check, leakage check).
        - "external": Questions that need competitor pricing, local/public events near the store, national market trends (inflation, oil prices, FRED), current weather for cities/regions, general web search queries, or client/customer lookup in the Salesforce CRM.
        - "workflow": Action requests (e.g. reordering items, updates, creating escalation tickets).
        - "conversational": Greetings, small talk, general questions unrelated to store operations.
        - "agentic": Complex queries that require checking multiple distinct sources of information (e.g. checking weather + inventory stock levels, comparing competitor prices + checking local catalog, events + sales trends).

        Available Database Schema context for deciding "db" queries:
        {DB_SCHEMA_REFERENCE}

        Classification Instructions:
        1. GREETINGS & CASUAL PREFIXES: If the user query contains greetings (e.g. "hi", "hello", "hey", "good morning") but ALSO contains an operational planning or lookup question (e.g. "wheather I can plan for...", "can you check...", "what is the stock..."), you MUST ignore the greeting and categorize the core operational request. Do NOT classify it as "conversational" if there is any operational request present!
        2. TYPO ROBUSTNESS: Treat common typos like "wheather" or "weather" as "whether" when introducing a question (e.g., "wheather i can plan..." -> "whether i can plan..."), "selled" as "sold", etc.
        3. COMPREHENSIVE AGENTIC PLANNING: Any request about whether the manager can plan sales or carry items (e.g., "can I plan for chicken noodles", "planning jacket sales") requires a comprehensive check of all operational tools. You MUST route these queries to the "agentic" category and construct a steps block that plans:
           - A SQLite Database step ("db") to check inventory levels (current stock, ROL, lead times).
           - A Weather forecast step ("external" with tool "weather") for the local store city to assess traffic impact.
           - A Competitor Pricing step ("external" with tool "pricing") to check convenience/grocery pricing via Apify.
           - A Market Trends step ("external" with tool "search" or "trends") to fetch general consumer trends or economic stats.
        4. PLURAL/SINGULAR ROBUSTNESS IN SQL: When generating SQL query steps in the steps block (under "query"), always handle both singular and plural forms for text search (e.g. use `(name LIKE '%chicken noodle%' OR name LIKE '%chicken noodles%')` or `(name LIKE '%jacket%' OR name LIKE '%jackets%')`) so matches succeed even if there is a singular/plural variation in the product name.
        5. STORE PROFILE & METRICS: Any questions asking for a store profile, store highlights, store performance overview, or dashboard KPI stats (e.g., "tell me the details about our store", "how is our store doing", "summarize store metrics") MUST be routed to the "db" category. This is because they require querying the database to aggregate live counts (such as total unique SKUs, total stock valuation, high-risk items, reorder limit counts, and overdue PO counts).
        6. SALESFORCE & CLIENT DETAILS: Any queries asking to search, lookup, retrieve, or list customer details or client contacts in Salesforce (e.g. "search John in Salesforce", "Who are the clients in our Salesforce customer database?", "Get details of client Kavin from Salesforce") MUST be routed to the "external" category. Do NOT route them to "rag" or "conversational".
        7. PROACTIVE REFILL & STOCKING RECOMMENDATIONS: Any query asking what products need to be refilled, restocked, or ordered (e.g., "which product is need to refill", "what should I refill today", "recommend products to stock up") requires a comprehensive multi-agent plan. You MUST route these queries to the "agentic" category and construct a steps block that plans:
           - A SQLite Database step ("db" with query "SELECT name, category, current_stock, rol, roq FROM inventory WHERE current_stock <= rol") to check low stock items.
           - An External step ("external" with tool "weather") for the local store city to assess weather-related demands.
           - An External step ("external" with tool "events") to check local events that might drive convenience store traffic.

        Available Database Schema context for deciding "db" queries:
        {DB_SCHEMA_REFERENCE}

        Provide the output in JSON format:
        {{
          "category": "db" | "rag" | "live" | "external" | "workflow" | "conversational" | "agentic",
          "plan": "One sentence describing how you will resolve this request.",
          "is_complex": true | false,
          "steps": [
             // ONLY populate this array if category is "agentic". List steps sequentially.
             // Each step has: 
             //   "agent": "db" | "rag" | "live" | "external" | "workflow"
             //   "query": "Strictly valid SQL query for 'db' agent (e.g. SELECT * FROM inventory WHERE (name LIKE '%chicken noodle%' OR name LIKE '%chicken noodles%')), or search query for RAG/workflow"
             //   "tool": "Optional. Specific tool for 'external' agent: 'weather' | 'pricing' | 'events' | 'trends' | 'search'"
             //   "param": "Optional. Parameter for external tool. For 'weather' and 'events', this MUST be strictly the city name (e.g., 'Chicago'). Do NOT include timeframe or temporal modifiers like 'today', 'tomorrow', 'next 2 days', or 'next week' inside 'param'."
          ]
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

    async def run_business_context_engine(self, query: str, category: str) -> dict:
        """Determines the business decisions, impact, and causal logic for a query."""
        logger.info(f"[BUSINESS CONTEXT ENGINE] Evaluating query: '{query}' under category: '{category}'")
        prompt = f"""
        You are the Business Context Engine of the AI Assistant.
        Your role is to apply professional store management and retail operations knowledge to evaluate a user's query BEFORE we generate the final recommendation.
        
        Analyze the query and output a JSON object with these exact fields:
        1. "influenced_decisions": List of specific store business decisions this query influences (e.g., ["Beverage demand", "Ice demand", "Staffing", "Inventory risk", "Pricing audit", "None"]).
        2. "causally_related": Boolean indicating if weather, event, or replenishment recommendations are causally related to this specific question.
           - Set to true ONLY if the user is asking about weather, upcoming events, vendor delays, product stockouts, refilling, ordering, or general planning.
           - Set to false if the user is asking about transactional database counts, metrics (e.g. "how many purchase transactions", "show sales volume"), customer CRM info (e.g. "find John Smith"), general policies, or greetings.
        3. "business_impact_template": A brief description of how this metric/topic impacts daily convenience retail operations.
        4. "action_guidelines": Actionable advice for the store manager based on the topic.
        
        Examples:
        - Query: "How's today's weather?"
          Output: {{
            "influenced_decisions": ["Beverage demand", "Ice demand", "Automotive fluids", "Seasonal products"],
            "causally_related": true,
            "business_impact_template": "Weather dictates immediate traffic patterns and category shifts (e.g., hot weather boosts beverage/water sales).",
            "action_guidelines": "Adjust stock presentation and verify cold-beverage levels."
          }}
          
        - Query: "How many purchase transactions?"
          Output: {{
            "influenced_decisions": ["Sales volume audit", "Customer traffic analysis"],
            "causally_related": false,
            "business_impact_template": "Transactional counts measure throughput and terminal utilization.",
            "action_guidelines": "Audit peak hours to optimize cashier scheduling."
          }}
          
        Current User Query: "{query}"
        Query Category: "{category}"
        
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
            logger.info(f"[BUSINESS CONTEXT ENGINE] Analysis: decisions={parsed.get('influenced_decisions')}, causally_related={parsed.get('causally_related')}")
            return parsed
        except Exception as e:
            logger.warning(f"[BUSINESS CONTEXT ENGINE] Error parsing business context: {e}. Defaulting.")
            return {
                "influenced_decisions": ["General Operations"],
                "causally_related": False,
                "business_impact_template": "Core metrics indicate store throughput and operations status.",
                "action_guidelines": "Monitor key performance indicators."
            }

    def _extract_days_from_query(self, query: str, default: int = 7) -> int:
        """Parses a query string to extract a timeframe in days."""
        q = (query or "").lower()
        import re
        # Check patterns like "next 3 days", "2 days", "5 days", "10 days"
        match = re.search(r'(\d+)\s*day', q)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        # Check patterns like "1 week", "2 weeks", "next week"
        if "1 week" in q or "next week" in q:
            return 7
        if "2 weeks" in q:
            return 14
        if "today" in q and "tomorrow" not in q and "week" not in q and "day" not in q:
            return 1
        if "tomorrow" in q and "week" not in q and "day" not in q:
            return 2
        return default

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
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === DB AGENT SQL TOOL RESULT ===")
            print(f"SQL: {cleaned_sql}")
            print(f"Returned {len(rows)} rows.")
            print(f"Headers: {headers}")
            print(f"Rows (first 5): {rows[:5]}")
            print(f"=======================================\n")
            
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

    def _restrict_to_north_america(self, param: str, user_city: str) -> Tuple[str, bool]:
        """Checks if the parameter is a city/country outside of North America and restricts it."""
        intl_keywords = [
            "london", "tokyo", "paris", "sydney", "berlin", "rome", "beijing", "seoul", 
            "uk", "japan", "france", "australia", "germany", "china", "india", "italy", 
            "europe", "asia", "england", "spain", "madrid", "singapore", "hong kong",
            "moscow", "mexico city", "rio", "cairo", "toronto", "vancouver", "montreal" # Wait, Toronto/Vancouver/Montreal are in Canada (North America)! So let's allow them!
        ]
        # We only block cities/countries outside North America (US and Canada).
        # Let's clean the blocklist to exclude North American locations.
        intl_blocklist = [
            "london", "tokyo", "paris", "sydney", "berlin", "rome", "beijing", "seoul", 
            "uk", "japan", "france", "australia", "germany", "china", "india", "italy", 
            "europe", "asia", "england", "spain", "madrid", "singapore", "hong kong",
            "moscow", "cairo", "brazil", "russia", "egypt", "africa", "south america"
        ]
        param_str = str(param).strip()
        param_lower = param_str.lower()
        
        for kw in intl_blocklist:
            if kw in param_lower:
                logger.warning(f"[SECURITY] Query param '{param_str}' attempted access to other countries. Restricting to North America (assigned store city: '{user_city}').")
                return user_city, True
                
        return param_str, False

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
        - CARTESIAN PRODUCT PREVENTION: NEVER JOIN multiple one-to-many tables (such as joining both `inventory` and `purchase_orders` or `sales` to `stores` in a single flat join) when using aggregates like SUM or COUNT. This causes row multiplication, inflating counts and valuations!
        - STORE OVERVIEW METRICS: If the user asks for store profile details and highlights (e.g. "tell me about our store", "summarize store metrics"), you MUST query the `stores` table s and calculate the metrics using independent subqueries in the SELECT clause, exactly like this:
          SELECT 
              s.id AS store_id, 
              s.name AS store_name, 
              s.city, 
              s.address, 
              s.contact, 
              s.active_since, 
              (SELECT COUNT(DISTINCT code) FROM inventory WHERE store_id = s.id) AS total_products, 
              (SELECT SUM(current_stock * unit_price) FROM inventory WHERE store_id = s.id) AS inventory_value, 
              (SELECT COUNT(*) FROM inventory WHERE store_id = s.id AND (current_stock * 1.0 / avg_daily_consumption) <= 7) AS critical_stockout_count, 
              (SELECT COUNT(*) FROM inventory WHERE store_id = s.id AND current_stock <= rol) AS below_reorder_count, 
              (SELECT COUNT(*) FROM purchase_orders WHERE store_id = s.id AND expected_delivery_date < date('now') AND status NOT IN ('Delivered', 'Cancelled')) AS overdue_orders_count
          FROM stores s 
          WHERE s.id = 'BP-CHI-1025'
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
           - Overdue Orders: Can be calculated as Count of purchase orders where `expected_delivery_date < date('now')` AND `status NOT IN ('Delivered', 'Cancelled')` (or products in inventory where `order_by_date < date('now')`). Use independent SELECT subqueries to compute these.
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

    def get_inventory_summary(self, store_id: str) -> dict:
        """Fetches a high-level summary of store inventory (low-stock items and category metrics)."""
        if not store_id:
            store_id = 'BP-CHI-1024'
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            # 1. Low stock items (where current_stock is less than or equal to reorder level ROL)
            cursor.execute("""
                SELECT name, category, current_stock, rol, roq, risk_level 
                FROM inventory 
                WHERE store_id = ? AND current_stock <= rol
            """, (store_id,))
            low_stock = [
                {
                    "name": r[0],
                    "category": r[1],
                    "current_stock": r[2],
                    "rol": r[3],
                    "roq": r[4],
                    "risk_level": r[5]
                }
                for r in cursor.fetchall()
            ]
            
            # 2. General category stocks
            cursor.execute("""
                SELECT category, COUNT(*), SUM(current_stock) 
                FROM inventory 
                WHERE store_id = ? 
                GROUP BY category
            """, (store_id,))
            categories = [
                {
                    "category": r[0],
                    "sku_count": r[1],
                    "total_stock": r[2]
                }
                for r in cursor.fetchall()
            ]
            conn.close()
            return {"low_stock": low_stock[:15], "categories": categories}
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Failed to fetch inventory summary: {e}")
            return {"low_stock": [], "categories": []}

    async def process(self, query: str, conversation: list, current_user: dict = None) -> dict:
        """Main orchestrator entrypoint."""
        store_id = current_user.get("storeId") if current_user else None
        if not store_id:
            store_id = "BP-CHI-1024"
        inventory_summary = self.get_inventory_summary(store_id)

        # Convert history format
        history_lines = []
        for c in conversation:
            if not c:
                continue
            parsed_dict = None
            if isinstance(c, str):
                c_stripped = c.strip()
                if (c_stripped.startswith("{") and c_stripped.endswith("}")) or (c_stripped.startswith("[") and c_stripped.endswith("]")):
                    try:
                        parsed_dict = json.loads(c)
                    except Exception:
                        pass
            elif isinstance(c, dict):
                parsed_dict = c

            if isinstance(parsed_dict, dict):
                if "user" in parsed_dict or "ai" in parsed_dict:
                    u = parsed_dict.get("user", "")
                    a = parsed_dict.get("ai", "")
                    if u or a:
                        history_lines.append(f"User: {u}\nAI: {a}")
                elif "role" in parsed_dict and "content" in parsed_dict:
                    role = str(parsed_dict.get("role", "")).lower()
                    content = parsed_dict.get("content", "")
                    if role == "user":
                        history_lines.append(f"User: {content}")
                    elif role in ["assistant", "ai"]:
                        history_lines.append(f"AI: {content}")
            else:
                c_str = str(c).strip()
                if c_str.lower().startswith("user:"):
                    history_lines.append(c_str)
                elif c_str.lower().startswith("ai:") or c_str.lower().startswith("ai :"):
                    suffix = c_str.split(":", 1)[1].strip()
                    history_lines.append(f"AI: {suffix}")
                elif c_str.lower().startswith("assistant:"):
                    suffix = c_str.split(":", 1)[1].strip()
                    history_lines.append(f"AI: {suffix}")
                else:
                    history_lines.append(c_str)
        history_text = "\n".join(history_lines)

        intent = await self.parse_intent(query, history_text)
        category = intent.get("category", "conversational")
        business_context = await self.run_business_context_engine(query, category)
        
        db_results = None
        rag_results = None
        live_results = None
        workflow_executed = None
        sql_generated = None
        external_results = None
        tools_used = []
        
        # 0. Agentic Multi-Step Planner
        if category == "agentic":
            steps = intent.get("steps", [])
            logger.info(f"[PLANNER AGENT] Starting sequential execution of {len(steps)} steps.")
            
            # Resolve user's city if storeId is provided
            user_city = "Chicago"
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
                except Exception as e:
                    logger.warning(f"[PLANNER AGENT] Failed to fetch city for store: {e}")

            db_results_list = []
            external_results_list = []
            rag_results_list = []
            live_results_list = []
            
            for step in steps:
                step_agent = step.get("agent")
                tool = step.get("tool", "")
                param = step.get("param", "") or step.get("query", "")
                
                # Normalize tool checks: redirect weather/pricing/events/trends to external
                if tool in ["weather", "pricing", "events", "trends", "search"]:
                    step_agent = "external"
                elif step_agent == "live" and any(w in str(param).lower() or w in str(tool).lower() for w in ["weather", "temp"]):
                    step_agent = "external"
                    tool = "weather"

                logger.info(f"[PLANNER AGENT] Executed step: agent={step_agent}, tool={tool}, param={param}")
                
                if step_agent == "db":
                    sql = step.get("query")
                    if sql:
                        sql = self._clean_sql_query(sql)
                        headers, rows, status = self.execute_db_query(sql)
                        if status == "success":
                            db_results_list.append({
                                "sql": sql,
                                "headers": headers,
                                "rows": [list(r) for r in rows]
                            })
                            tools_used.append("Internal Database")
                
                elif step_agent == "external":
                    param, overridden = self._restrict_to_north_america(param, user_city)
                    if tool == "weather":
                        if not param or len(str(param).split()) > 2 or any(w in str(param).lower() for w in ["local", "current", "today", "forecast", "get"]):
                            param = user_city
                        from app.external_services.weather_service import get_weather
                        try:
                            res = await get_weather(param)
                            external_results_list.append(f"Weather forecast for {param} (North America): {res}")
                            tools_used.append("Weather Tool")
                        except Exception as e:
                            logger.error(f"Weather API step failed: {e}")
                    elif tool == "pricing":
                        if not param or len(str(param).split()) > 4:
                            param = "Convenience retail items pricing"
                        search_param = f"{param} in USA convenience stores"
                        from app.external_services.competitor_pricing import get_competitor_pricing
                        try:
                            res = await get_competitor_pricing(search_param)
                            external_results_list.append(f"Competitor pricing for {param} (North America): {res}")
                            tools_used.append("Competitor Pricing Tool")
                        except Exception as e:
                            logger.error(f"Pricing API step failed: {e}")
                    elif tool == "events":
                        if not param or len(str(param).split()) > 2:
                            param = user_city
                        from app.external_services.event_tracker import get_local_events
                        try:
                            days_param = self._extract_days_from_query(query)
                            res = await get_local_events(param, days=days_param)
                            external_results_list.append(f"Events near {param} (North America): {res}")
                            tools_used.append("Events Tool")
                        except Exception as e:
                            logger.error(f"Events API step failed: {e}")
                    elif tool == "trends":
                        if not param or len(str(param).split()) > 2:
                            param = "inflation"
                        # Make sure to query standard US series (e.g. mapping metric to US)
                        from app.external_services.market_trends import get_market_trend
                        try:
                            res = await get_market_trend(param)
                            external_results_list.append(f"Economic trends for {param} (North America): {res}")
                            tools_used.append("Economic Trends Tool")
                        except Exception as e:
                            logger.error(f"Trends API step failed: {e}")
                    else:
                        search_param = f"{param} retail market North America"
                        from app.external_services.web_search import search_web
                        try:
                            res = await search_web(search_param)
                            external_results_list.append(f"Web search for {param} (North America): {res}")
                            tools_used.append("Web Search Tool")
                        except Exception as e:
                            logger.error(f"Search API step failed: {e}")
                
                elif step_agent == "rag":
                    rag_docs = self.rag.search(step.get("query", query), top_k=2)
                    for r in rag_docs:
                        rag_results_list.append({
                            "title": r["doc"]["title"],
                            "content": r["doc"]["content"]
                        })
                    tools_used.append("SOP Guidelines")
                
                elif step_agent == "live":
                    live_results_list.append(self.get_live_sensor_readings())
                    tools_used.append("IoT Sensors")
            
            # Combine all results
            if db_results_list:
                db_results = {
                    "headers": db_results_list[0]["headers"],
                    "rows": db_results_list[0]["rows"]
                }
            if external_results_list:
                external_results = "\n\n".join(external_results_list)
            if rag_results_list:
                rag_results = rag_results_list
            if live_results_list:
                live_results = live_results_list[0]

        # 1. DB Agent
        elif category == "db":
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
                tools_used.append("Internal Database")
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
            tools_used.append("SOP Guidelines")
            logger.info(f"[RAG AGENT] Complete. Retrieved {len(rag_results)} document(s).")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === RAG AGENT TOOL RESULT ===")
            print(f"RAG Query: {query}")
            print(f"Documents found: {len(rag_results)}")
            for idx, doc in enumerate(rag_results, 1):
                print(f"  [{idx}] Title: {doc['title']} (Score: {doc['score']:.4f})")
                print(f"      Content Preview: {doc['content'][:150]}...")
            print(f"========================================\n")
 
        # 3. Live IoT Agent
        elif category == "live":
            logger.info("[LIVE AGENT] Fetching real-time IoT sensor readings.")
            live_results = self.get_live_sensor_readings()
            logger.info(f"[LIVE AGENT] Retrieved {len(live_results.get('sensors', []))} sensor reading(s).")
            tools_used.append("IoT Sensors")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === LIVE IoT AGENT TOOL RESULT ===")
            print(f"IoT Sensors: {live_results.get('sensors')}")
            print(f"Local Weather: {live_results.get('weather')}")
            print(f"=============================================\n")

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
            - "salesforce": Use this when the query requires searching, listing, retrieving, or looking up customer/client details or CRM records from the Salesforce system (e.g. searching client lists, customer contacts).

            Provide output in JSON format:
            {{
                "tool": "search" | "weather" | "pricing" | "events" | "trends" | "salesforce",
                "param": "The key parameter (e.g. search query, city name, product name, or economic metric type like 'inflation')."
            }}

            CRITICAL PARAMETER CLEANING RULE:
            - For "weather" and "events" tools, the "param" MUST be strictly the city name or location name (e.g., "Chicago", "Denver").
            - Do NOT include temporal modifiers or relative timeframes (such as "today", "tomorrow", "next 2 days", "next week", "for 1 week") in the "param" string. Return ONLY the location name itself.
            - Example: "events in Chicago for next 2 days" -> tool: "events", param: "Chicago".

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
                    tools_used.append("Web Search Tool")
                elif tool == "weather":
                    from app.external_services.weather_service import get_weather
                    external_results = await get_weather(param)
                    tools_used.append("Weather Tool")
                elif tool == "pricing":
                    from app.external_services.competitor_pricing import get_competitor_pricing
                    external_results = await get_competitor_pricing(param)
                    tools_used.append("Competitor Pricing Tool")
                elif tool == "events":
                    from app.external_services.event_tracker import get_local_events
                    days_param = self._extract_days_from_query(query)
                    external_results = await get_local_events(param, days=days_param)
                    tools_used.append("Events Tool")
                elif tool == "trends":
                    from app.external_services.market_trends import get_market_trend
                    external_results = await get_market_trend(param)
                    tools_used.append("Economic Trends Tool")
                elif tool == "salesforce":
                    from app.ai_analyzer.tools import query_salesforce_customer
                    res_content, usage = await query_salesforce_customer(param, conversation=conversation)
                    external_results = res_content
                    tools_used.append("Customer CRM Tool")
                else:
                    external_results = "Unknown tool requested."
                
                # Print the tool results directly to the terminal for tracking
                print(f"\n[TRACKING] === EXTERNAL TOOL RESULT ({tool}) ===")
                print(f"Parameter: {param}")
                print(f"Result Preview: {str(external_results)[:300]}...")
                print(f"================================================\n")
                
            except Exception as e:
                logger.warning(f"[EXTERNAL AGENT] Classification failed: {e}. Defaulting to Tavily search.")
                from app.external_services.web_search import search_web
                external_results = await search_web(query)
                tools_used.append("Web Search Tool")
                
                # Print the tool results directly to the terminal for tracking
                print(f"\n[TRACKING] === EXTERNAL TOOL RESULT (search-fallback) ===")
                print(f"Query: {query}")
                print(f"Result Preview: {str(external_results)[:300]}...")
                print(f"=========================================================\n")

        # 5. Workflow Agent
        elif category == "workflow":
            logger.info("[WORKFLOW AGENT] Executing workflow action.")
            if "reorder" in query.lower() or "order" in query.lower():
                workflow_executed = "Draft Purchase Order created. Assigned ID 'PO-99120'. Items matched to reorder quantities (ROQ). Final manager signature required in ERP/SAP."
            else:
                workflow_executed = "Escalation ticket generated: 'Vendor Delivery Issue VND-001' logged in operations log under ID 'TKT-82190'."
            logger.info(f"[WORKFLOW AGENT] Result: {workflow_executed}")
            tools_used.append("Workflow Tool")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === WORKFLOW AGENT TOOL RESULT ===")
            print(f"Workflow Action Executed: {workflow_executed}")
            print(f"=============================================\n")

        # Synthesis Agent
        logger.info(f"[SYNTHESIS AGENT] Composing final response using model: {self.active_model}")
        synthesis_prompt = f"""
        You are the **AI Assistant** — a conversational, multi-level Agentic AI System primarily focused on supporting BP store managers, operations advisors, and vendor managers.
        You are capable of: Answer, Analyze, Compare, Predict, Recommend, Execute, Monitor, and Notify.
        Respond to the user's store operations query based on the fetched context, execution details, and business context analysis.
        
        User Query: "{query}"
        Category Routed: {category}
        
        Context Details:
        - DB Execution Results: {json.dumps(db_results) if db_results else "None"}
        - RAG Retrieval: {json.dumps(rag_results) if rag_results else "None"}
        - Live IoT Sensors: {json.dumps(live_results) if live_results else "None"}
        - Workflow Executed: {workflow_executed if workflow_executed else "None"}
        - External Web Context: {external_results if external_results else "None"}
        - Live Store Inventory Summary (Low Stock & Category Health): {json.dumps(inventory_summary) if inventory_summary else "None"}
        
        Business Context Engine Analysis:
        {json.dumps(business_context)}

        ════════════════════════════════════════════════════════════
        MANDATORY RESPONSE FORMAT — YOU MUST FOLLOW THIS STRUCTURE EXACTLY
        ════════════════════════════════════════════════════════════

        Always structure your response using these FOUR parts in this exact order:

        **Requested Information / Summary**
        Provide the direct, precise answer to the user's query.
        - Use structured layouts for pricing comparisons, market trends, or supply chain checks (see rules below).
        - Format other tabular database records as a clean Markdown table in this section.
        - Keep weather or event queries overview extremely brief.

        **Business Impact**
        Provide a list of at most 1 or 2 high-level bullet points detailing the most critical business implications. Focus strictly on what matters.
        For each bullet point, write a bold category/operational label, followed by a colon and a very short explanation (max 10–12 words max) explaining *how* and *why* this area is impacted based on the current context.

        **Recommended Actions**
        Provide clear, prioritized next steps. Group them into logical subheadings:
        
        1. **Restocking Priorities**:
           - Suggest restocking priorities in a single, high-level bullet point.
           - If low-stock items are explicitly relevant to the query context, list them using a short Markdown Table (max 3 items):
             | Product | Category | Current Stock | ROL | Contextual Suggestion |
             Each "Contextual Suggestion" must be a very short sentence (max 10 words) connecting low stock to the query context.
             
        2. **Displays & Positioning** (ONLY include if weather or events demand changes; otherwise, omit entirely):
           - Suggest a single high-level display adjustment in a single short bullet point (max 12 words). Do NOT mention any operational hours or timings.
           
        3. **Staffing** (ONLY include if weather or events demand changes; otherwise, omit entirely):
           - Suggest a single high-level staffing recommendation in a single short bullet point (max 12 words). Do NOT mention any operational hours or timings.

        *[Close with a single italicised call-to-action question dynamically tailored to the user's specific query and response content.]*

        ════════════════════════════════════════════════════════════
        ADDITIONAL RULES (apply on top of the format above)
        ════════════════════════════════════════════════════════════
        - CONFLICT RESOLUTION: Always use exact numbers from 'DB Execution Results'. Never alter or guess figures.
        - Do NOT show SQL code or reference internal system details.
        - Do NOT reference "right panel", "Analytics Panel", or any external dashboard.
        - Do NOT wrap the response in code fences.
        - SECTION HEADERS: Use bold markdown (`**Section Name**`) — never use `###` headings for section titles.
        - CONCISENESS: Keep the entire response extremely brief (typically under 150-250 words total). Never write more than 2 consecutive sentences of plain paragraph text. Ensure all bullet points and table cell entries are short, direct, and punchy.
        - NO TIMINGS: Never mention specific timings, hours, or timeframes for staffing or positioning. Focus strictly on high-level recommendations.
        - NO SPECULATION: Unless the provided database results or tool context explicitly contain specific numbers or percentages, NEVER invent or speculate percentages/statistics (e.g. do NOT invent "ice sales up 20%"). Use qualitative terms like "expected to increase demand for cold beverages" instead.
        - COMPETITOR PRICING COMPARISON: Format pricing comparisons exactly like this structure instead of using prose:
          Your Price: $[Your price]
          [Competitor Name]: $[Competitor price]
          Difference: [+$Diff / -$Diff]
          Recommendation: [Maintain price / Reduce to $X.XX / etc.]
        - MARKET TRENDS: Never output generic statements like "CPI increased". State the exact change (e.g. "CPI increased 0.4%") and link it directly to a store operational action (e.g. "This may increase wholesale grocery costs. Monitor pricing for milk, bread, and beverages").
        - SUPPLY CHAIN DISRUPTIONS: If no disruptions are found, do NOT say generic phrases like "No supply chain issues". Format the response as:
          I checked:
          - Vendor purchase orders
          - News sources
          - Product recalls
          No active supply chain disruption was found for your current inventory.
        - CAUSAL RULE: If "causally_related" is FALSE in the Business Context Engine block, keep Recommended Actions strictly focused on the query topic. Do NOT append weather/event/low-stock refill recommendations.
        - WEATHER QUERIES (causally_related=true): If hot (>22°C) or clear, recommend cold beverages, water, ice. If cold (<10°C) or rainy/snowy, recommend hot drinks, food, or automotive antifreeze.
        - EVENT QUERIES (causally_related=true): High-attendance events drive grab-and-go demand (energy drinks, snacks, sandwiches).
        - SENSOR WARNINGS: If IoT sensor readings exceed safe limits, flag them with a Warning label.
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
        elif (category == "external" or category == "agentic") and external_results:
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
            "token_usage": token_usage,
            "tools_used": tools_used
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
        store_id = current_user.get("storeId") if current_user else None
        if not store_id:
            store_id = "BP-CHI-1024"
        inventory_summary = self.get_inventory_summary(store_id)

        # 1. Parse history
        history_lines = []
        for c in conversation:
            if not c:
                continue
            parsed_dict = None
            if isinstance(c, str):
                c_stripped = c.strip()
                if (c_stripped.startswith("{") and c_stripped.endswith("}")) or (c_stripped.startswith("[") and c_stripped.endswith("]")):
                    try:
                        parsed_dict = json.loads(c)
                    except Exception:
                        pass
            elif isinstance(c, dict):
                parsed_dict = c

            if isinstance(parsed_dict, dict):
                if "user" in parsed_dict or "ai" in parsed_dict:
                    u = parsed_dict.get("user", "")
                    a = parsed_dict.get("ai", "")
                    if u or a:
                        history_lines.append(f"User: {u}\nAI: {a}")
                elif "role" in parsed_dict and "content" in parsed_dict:
                    role = str(parsed_dict.get("role", "")).lower()
                    content = parsed_dict.get("content", "")
                    if role == "user":
                        history_lines.append(f"User: {content}")
                    elif role in ["assistant", "ai"]:
                        history_lines.append(f"AI: {content}")
            else:
                c_str = str(c).strip()
                if c_str.lower().startswith("user:"):
                    history_lines.append(c_str)
                elif c_str.lower().startswith("ai:") or c_str.lower().startswith("ai :"):
                    suffix = c_str.split(":", 1)[1].strip()
                    history_lines.append(f"AI: {suffix}")
                elif c_str.lower().startswith("assistant:"):
                    suffix = c_str.split(":", 1)[1].strip()
                    history_lines.append(f"AI: {suffix}")
                else:
                    history_lines.append(c_str)
        history_text = "\n".join(history_lines)

        # Yield initial intent parsing step
        yield json.dumps({"type": "step", "message": "Analyzing query intent..."})

        intent = await self.parse_intent(query, history_text)
        category = intent.get("category", "conversational")
        
        yield json.dumps({"type": "step", "message": "Enriching business context..."})
        business_context = await self.run_business_context_engine(query, category)
        
        db_results = None
        rag_results = None
        live_results = None
        workflow_executed = None
        sql_generated = None
        external_results = None
        tools_used = []

        # Yield routed step
        yield json.dumps({"type": "step", "message": f"Query routed to category: '{category}'"})

        # 0. Agentic Multi-Step Planner
        if category == "agentic":
            steps = intent.get("steps", [])
            yield json.dumps({"type": "step", "message": f"Planner decided on {len(steps)} steps: {intent.get('plan','')}"})
            
            # Resolve user's city if storeId is provided
            user_city = "Chicago"
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
                except Exception as e:
                    logger.warning(f"[PLANNER AGENT] Failed to fetch city for store in stream: {e}")

            db_results_list = []
            external_results_list = []
            rag_results_list = []
            live_results_list = []
            
            for idx, step in enumerate(steps, 1):
                step_agent = step.get("agent")
                tool = step.get("tool", "")
                param = step.get("param", "") or step.get("query", "")
                
                # Normalize tool checks: redirect weather/pricing/events/trends to external
                if tool in ["weather", "pricing", "events", "trends", "search"]:
                    step_agent = "external"
                elif step_agent == "live" and any(w in str(param).lower() or w in str(tool).lower() for w in ["weather", "temp"]):
                    step_agent = "external"
                    tool = "weather"

                # Yield progress updates to user
                msg = f"[Step {idx}/{len(steps)}] Executing {step_agent}"
                if tool:
                    msg += f" ({tool})"
                yield json.dumps({"type": "step", "message": msg})
                
                if step_agent == "db":
                    sql = step.get("query")
                    if sql:
                        sql = self._clean_sql_query(sql)
                        headers, rows, status = self.execute_db_query(sql)
                        if status == "success":
                            db_results_list.append({
                                "sql": sql,
                                "headers": headers,
                                "rows": [list(r) for r in rows]
                            })
                            tools_used.append("Internal Database")
                
                elif step_agent == "external":
                    param, overridden = self._restrict_to_north_america(param, user_city)
                    if tool == "weather":
                        if not param or len(str(param).split()) > 2 or any(w in str(param).lower() for w in ["local", "current", "today", "forecast", "get"]):
                            param = user_city
                        from app.external_services.weather_service import get_weather
                        try:
                            res = await get_weather(param)
                            external_results_list.append(f"Weather forecast for {param} (North America): {res}")
                            tools_used.append("Weather Tool")
                        except Exception as e:
                            logger.error(f"Weather API stream failed: {e}")
                    elif tool == "pricing":
                        if not param or len(str(param).split()) > 4:
                            param = "Convenience retail items pricing"
                        search_param = f"{param} in USA convenience stores"
                        from app.external_services.competitor_pricing import get_competitor_pricing
                        try:
                            res = await get_competitor_pricing(search_param)
                            external_results_list.append(f"Competitor pricing for {param} (North America): {res}")
                            tools_used.append("Competitor Pricing Tool")
                        except Exception as e:
                            logger.error(f"Pricing API stream failed: {e}")
                    elif tool == "events":
                        if not param or len(str(param).split()) > 2:
                            param = user_city
                        from app.external_services.event_tracker import get_local_events
                        try:
                            days_param = self._extract_days_from_query(query)
                            res = await get_local_events(param, days=days_param)
                            external_results_list.append(f"Events near {param} (North America): {res}")
                            tools_used.append("Events Tool")
                        except Exception as e:
                            logger.error(f"Events API stream failed: {e}")
                    elif tool == "trends":
                        if not param or len(str(param).split()) > 2:
                            param = "inflation"
                        from app.external_services.market_trends import get_market_trend
                        try:
                            res = await get_market_trend(param)
                            external_results_list.append(f"Economic trends for {param} (North America): {res}")
                            tools_used.append("Economic Trends Tool")
                        except Exception as e:
                            logger.error(f"Trends API stream failed: {e}")
                    else:
                        search_param = f"{param} retail market North America"
                        from app.external_services.web_search import search_web
                        try:
                            res = await search_web(search_param)
                            external_results_list.append(f"Web search for {param} (North America): {res}")
                            tools_used.append("Web Search Tool")
                        except Exception as e:
                            logger.error(f"Search API stream failed: {e}")
                
                elif step_agent == "rag":
                    rag_docs = self.rag.search(step.get("query", query), top_k=2)
                    for r in rag_docs:
                        rag_results_list.append({
                            "title": r["doc"]["title"],
                            "content": r["doc"]["content"]
                        })
                    tools_used.append("SOP Guidelines")
                
                elif step_agent == "live":
                    live_results_list.append(self.get_live_sensor_readings())
                    tools_used.append("IoT Sensors")
            
            # Combine all results
            if db_results_list:
                db_results = {
                    "headers": db_results_list[0]["headers"],
                    "rows": db_results_list[0]["rows"]
                }
            if external_results_list:
                external_results = "\n\n".join(external_results_list)
            if rag_results_list:
                rag_results = rag_results_list
            if live_results_list:
                live_results = live_results_list[0]

        # 1. DB Agent
        elif category == "db":
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
                tools_used.append("Internal Database")
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
            tools_used.append("SOP Guidelines")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === RAG AGENT TOOL RESULT (STREAM) ===")
            print(f"RAG Query: {query}")
            print(f"Documents found: {len(rag_results)}")
            for idx, doc in enumerate(rag_results, 1):
                print(f"  [{idx}] Title: {doc['title']} (Score: {doc['score']:.4f})")
                print(f"      Content Preview: {doc['content'][:150]}...")
            print(f"=================================================\n")
 
        # Live IoT Agent
        elif category == "live":
            yield json.dumps({"type": "step", "message": "Reading real-time IoT sensors..."})
            live_results = self.get_live_sensor_readings()
            tools_used.append("IoT Sensors")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === LIVE IoT AGENT TOOL RESULT (STREAM) ===")
            print(f"IoT Sensors: {live_results.get('sensors')}")
            print(f"Local Weather: {live_results.get('weather')}")
            print(f"======================================================\n")

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
            - "salesforce": Use this when the query requires searching, listing, retrieving, or looking up customer/client details or CRM records from the Salesforce system (e.g. searching client lists, customer contacts).

            Provide output in JSON format:
            {{
                "tool": "search" | "weather" | "pricing" | "events" | "trends" | "salesforce",
                "param": "The key parameter (e.g. search query, city name, product name, or economic metric type like 'inflation')."
            }}

            CRITICAL PARAMETER CLEANING RULE:
            - For "weather" and "events" tools, the "param" MUST be strictly the city name or location name (e.g., "Chicago", "Denver").
            - Do NOT include temporal modifiers or relative timeframes (such as "today", "tomorrow", "next 2 days", "next week", "for 1 week") in the "param" string. Return ONLY the location name itself.
            - Example: "events in Chicago for next 2 days" -> tool: "events", param: "Chicago".

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
                    tools_used.append("Web Search Tool")
                elif tool == "weather":
                    yield json.dumps({"type": "step", "message": f"Fetching weather forecast for: '{param}'..."})
                    from app.external_services.weather_service import get_weather
                    external_results = await get_weather(param)
                    tools_used.append("Weather Tool")
                elif tool == "pricing":
                    yield json.dumps({"type": "step", "message": f"Querying competitor pricing for: '{param}'..."})
                    from app.external_services.competitor_pricing import get_competitor_pricing
                    external_results = await get_competitor_pricing(param)
                    tools_used.append("Competitor Pricing Tool")
                elif tool == "events":
                    yield json.dumps({"type": "step", "message": f"Checking local events near: '{param}'..."})
                    from app.external_services.event_tracker import get_local_events
                    days_param = self._extract_days_from_query(query)
                    external_results = await get_local_events(param, days=days_param)
                    tools_used.append("Events Tool")
                elif tool == "trends":
                    yield json.dumps({"type": "step", "message": f"Querying FRED market index for: '{param}'..."})
                    from app.external_services.market_trends import get_market_trend
                    external_results = await get_market_trend(param)
                    tools_used.append("Economic Trends Tool")
                elif tool == "salesforce":
                    yield json.dumps({"type": "step", "message": f"Querying Salesforce customer database for: '{param}'..."})
                    from app.ai_analyzer.tools import query_salesforce_customer
                    res_content, usage = await query_salesforce_customer(param, conversation=conversation)
                    external_results = res_content
                    tools_used.append("Customer CRM Tool")
                
                # Print the tool results directly to the terminal for tracking
                print(f"\n[TRACKING] === EXTERNAL TOOL RESULT (STREAM - {tool}) ===")
                print(f"Parameter: {param}")
                print(f"Result Preview: {str(external_results)[:300]}...")
                print(f"=========================================================\n")
                
            except Exception:
                yield json.dumps({"type": "step", "message": "Searching web..."})
                from app.external_services.web_search import search_web
                external_results = await search_web(query)
                tools_used.append("Web Search Tool")
                
                # Print the tool results directly to the terminal for tracking
                print(f"\n[TRACKING] === EXTERNAL TOOL RESULT (STREAM - search-fallback) ===")
                print(f"Query: {query}")
                print(f"Result Preview: {str(external_results)[:300]}...")
                print(f"==================================================================\n")

        # Workflow Agent
        elif category == "workflow":
            yield json.dumps({"type": "step", "message": "Logging workflow transaction..."})
            if "reorder" in query.lower() or "order" in query.lower():
                workflow_executed = "Draft Purchase Order created. Assigned ID 'PO-99120'. Items matched to reorder quantities (ROQ). Final manager signature required in ERP/SAP."
            else:
                workflow_executed = "Escalation ticket generated: 'Vendor Delivery Issue VND-001' logged in operations log under ID 'TKT-82190'."
            tools_used.append("Workflow Tool")
            
            # Print the tool results directly to the terminal for tracking
            print(f"\n[TRACKING] === WORKFLOW AGENT TOOL RESULT (STREAM) ===")
            print(f"Workflow Action Executed: {workflow_executed}")
            print(f"=====================================================\n")

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
        You are the **AI Assistant** — a conversational, multi-level Agentic AI System primarily focused on supporting BP store managers, operations advisors, and vendor managers.
        You are capable of: Answer, Analyze, Compare, Predict, Recommend, Execute, Monitor, and Notify.
        Respond to the user's store operations query based on the fetched context, history context, execution details, and business context analysis.
        
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
        - Live Store Inventory Summary (Low Stock & Category Health): {json.dumps(inventory_summary) if inventory_summary else "None"}
        
        Business Context Engine Analysis:
        {json.dumps(business_context)}

        ════════════════════════════════════════════════════════════
        MANDATORY RESPONSE FORMAT — YOU MUST FOLLOW THIS STRUCTURE EXACTLY
        ════════════════════════════════════════════════════════════

        Always structure your response using these FOUR parts in this exact order:

        **Requested Information / Summary**
        Provide the direct, precise answer to the user's query.
        - Use structured layouts for pricing comparisons, market trends, or supply chain checks (see rules below).
        - Format other tabular database records as a clean Markdown table in this section.
        - Keep weather or event queries overview extremely brief.

        **Business Impact**
        Provide a list of at most 1 or 2 high-level bullet points detailing the most critical business implications. Focus strictly on what matters.
        For each bullet point, write a bold category/operational label, followed by a colon and a very short explanation (max 10–12 words max) explaining *how* and *why* this area is impacted based on the current context.

        **Recommended Actions**
        Provide clear, prioritized next steps. Group them into logical subheadings:
        
        1. **Restocking Priorities**:
           - Suggest restocking priorities in a single, high-level bullet point.
           - If low-stock items are explicitly relevant to the query context, list them using a short Markdown Table (max 3 items):
             | Product | Category | Current Stock | ROL | Contextual Suggestion |
             Each "Contextual Suggestion" must be a very short sentence (max 10 words) connecting low stock to the query context.
             
        2. **Displays & Positioning** (ONLY include if weather or events demand changes; otherwise, omit entirely):
           - Suggest a single high-level display adjustment in a single short bullet point (max 12 words). Do NOT mention any operational hours or timings.
           
        3. **Staffing** (ONLY include if weather or events demand changes; otherwise, omit entirely):
           - Suggest a single high-level staffing recommendation in a single short bullet point (max 12 words). Do NOT mention any operational hours or timings.

        *[Close with a single italicised call-to-action question dynamically tailored to the user's specific query and response content.]*

        ════════════════════════════════════════════════════════════
        ADDITIONAL RULES (apply on top of the format above)
        ════════════════════════════════════════════════════════════
        - CONFLICT RESOLUTION: Always use exact numbers from 'DB Execution Results'. Never alter or guess figures.
        - Do NOT show SQL code or reference internal system details.
        - Do NOT reference "right panel", "Analytics Panel", or any external dashboard.
        - Do NOT wrap the response in code fences.
        - SECTION HEADERS: Use bold markdown (`**Section Name**`) — never use `###` headings for section titles.
        - CONCISENESS: Keep the entire response extremely brief (typically under 150-250 words total). Never write more than 2 consecutive sentences of plain paragraph text. Ensure all bullet points and table cell entries are short, direct, and punchy.
        - NO TIMINGS: Never mention specific timings, hours, or timeframes for staffing or positioning. Focus strictly on high-level recommendations.
        - NO SPECULATION: Unless the provided database results or tool context explicitly contain specific numbers or percentages, NEVER invent or speculate percentages/statistics (e.g. do NOT invent "ice sales up 20%"). Use qualitative terms like "expected to increase demand for cold beverages" instead.
        - COMPETITOR PRICING COMPARISON: Format pricing comparisons exactly like this structure instead of using prose:
          Your Price: $[Your price]
          [Competitor Name]: $[Competitor price]
          Difference: [+$Diff / -$Diff]
          Recommendation: [Maintain price / Reduce to $X.XX / etc.]
        - MARKET TRENDS: Never output generic statements like "CPI increased". State the exact change (e.g. "CPI increased 0.4%") and link it directly to a store operational action (e.g. "This may increase wholesale grocery costs. Monitor pricing for milk, bread, and beverages").
        - SUPPLY CHAIN DISRUPTIONS: If no disruptions are found, do NOT say generic phrases like "No supply chain issues". Format the response as:
          I checked:
          - Vendor purchase orders
          - News sources
          - Product recalls
          No active supply chain disruption was found for your current inventory.
        - CAUSAL RULE: If "causally_related" is FALSE in the Business Context Engine block, keep Recommended Actions strictly focused on the query topic. Do NOT append weather/event/low-stock refill recommendations.
        - WEATHER QUERIES (causally_related=true): If hot (>22°C) or clear, recommend cold beverages, water, ice. If cold (<10°C) or rainy/snowy, recommend hot drinks, food, or automotive antifreeze.
        - EVENT QUERIES (causally_related=true): High-attendance events drive grab-and-go demand (energy drinks, snacks, sandwiches).
        - SENSOR WARNINGS: If IoT sensor readings exceed safe limits, flag them with a Warning label.
        """

        full_text = ""
        async for chunk in self.llm.astream(synthesis_prompt):
            content = chunk.content
            full_text += content
            yield json.dumps({"type": "token", "text": content})

        prompt_tokens = max(1, int(len(synthesis_prompt) / 3.7))
        completion_tokens = max(1, int(len(full_text) / 3.7))
        total_tokens = prompt_tokens + completion_tokens

        logger.info(f"[MODEL] Model used: {self.active_model}")
        logger.info(f"[TOKENS] Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")

        visualization = None
        report_markdown = None
        
        if table_data:
            visualization = self._auto_visualize(query, db_results)
        elif rag_results:
            report_markdown = full_text
        elif (category == "external" or category == "agentic") and external_results:
            report_markdown = f"# External Query Source Details\n\nQuery: {query}\n\n{external_results}"

        yield json.dumps({
            "type": "result",
            "tableData": table_data,
            "chartData": visualization,
            "reportData": report_markdown,
            "tools_used": tools_used,
            "model_name": self.active_model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        })
