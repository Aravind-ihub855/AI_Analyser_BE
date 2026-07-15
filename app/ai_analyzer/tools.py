import pandas as pd
import os
import json
import pandas as pd
import os
import json
from langchain_experimental.tools import PythonREPLTool
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3
import re
import logging
from typing import Tuple, Dict, Any

from app.services.llm import get_groq_llm
llm = get_groq_llm()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global DB path (will be set by agent_Main.py)
current_db_path = None

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


def clean_sql_query(sql_response: str) -> str:
    """Clean SQL response: remove prefixes like 'sql ', 'SQL:', 'Query:', and code wrappers."""
    cleaned = re.sub(r'^(sql\s+|query\s*:?\s*|select\s*query\s*:?\s*)', '', sql_response, flags=re.IGNORECASE)
    cleaned = cleaned.strip('`').strip().strip(';').strip()  # Remove wrappers and extra ;
    return cleaned

def clean_column_name(col: str) -> str:
    """Normalize column names for SQLite safety."""
    col = col.strip().lower()
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)    # replace spaces & symbols with _
    col = re.sub(r"_+", "_", col)               # collapse __ → _
    col = col.strip("_")                        # remove leading/trailing _
    return col

def force_data_table(sql: str) -> str:
    """
    Replace any hallucinated table name with 'data'.
    IMPORTANT: Preserves CTEs and their references - only replaces actual table names, not CTE names.
    Handles FROM/JOIN statements carefully.
    Also normalizes column names in SQL to be snake_case if they are quoted.
    """
    import re
    
    # 1. Normalize quoted column names (e.g. "Order Date" -> "order_date")
    def normalize_col(match):
        col_name = match.group(1)
        return f'"{clean_column_name(col_name)}"'
    
    # Replace "Column Name" or `Column Name` with normalized version
    sql = re.sub(r'["`]([^"`]+)["`]', normalize_col, sql)

    # 2. Handle Table Names (CTE aware)
    # Check if there's a CTE (WITH ... AS) in the SQL
    has_cte = re.search(r"\bWITH\s+\w+\s+AS\s*\(", sql, flags=re.IGNORECASE)
    
    if has_cte:
        # If there's a CTE, only replace table names in FROM/JOIN that are NOT preceded by WITH
        # Extract CTE names to avoid replacing them
        cte_names = re.findall(r"\bWITH\s+(\w+)\s+AS|,\s*(\w+)\s+AS\s*\(", sql, flags=re.IGNORECASE)
        cte_names_list = [name for pair in cte_names for name in pair if name]
        
        # Replace table names in FROM/JOIN, but NOT if they're CTE names
        def replace_table_name(match):
            prefix = match.group(1)  # FROM or JOIN
            table_name = match.group(2)  # table name
            
            # If it's a CTE name, keep it
            if table_name.lower() in [cte.lower() for cte in cte_names_list]:
                return f"{prefix} {table_name}"
            # Otherwise, it's likely a hallucinated table name -> replace with 'data'
            else:
                return f"{prefix} data"
        
        sql = re.sub(r"(?i)\b(FROM|JOIN)\s+[`\"]?(\w+)[`\"]?", replace_table_name, sql)
    else:
        # No CTE, aggressively replace any table name in FROM/JOIN with 'data'
        sql = re.sub(r"(?i)\bFROM\s+[`\"]?(\w+)[`\"]?", "FROM data", sql)
        sql = re.sub(r"(?i)\bJOIN\s+[`\"]?(\w+)[`\"]?", "JOIN data", sql)
    
    return sql


def init_db_from_dataframe(df: pd.DataFrame, db_path: str) -> None:
    """Initialize SQLite DB from an in-memory pandas DataFrame."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Clean all column names to be database-friendly
    df.columns = [clean_column_name(col) for col in df.columns]

    # Save the cleaned DataFrame to the SQLite database
    df.to_sql('data', conn, if_exists='replace', index=False, method='multi')

    # Create indexes on numeric columns for faster queries
    for col in df.columns:
        safe_col_name = clean_column_name(col)
        if safe_col_name and df[col].dtype in ['int64', 'float64']:
            try:
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{safe_col_name} ON data("{safe_col_name}")')
            except Exception as e:
                logger.warning(f"Could not create index on column {col}: {e}")

    conn.commit()
    conn.close()


def get_columns_list(db_path: str) -> list:
    """Get list of actual column names from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(data)")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        return columns
    except Exception as e:
        logger.error(f"Error retrieving columns: {str(e)}")
        return []

def get_schema_and_sample(db_path: str, sample_size: int = 20) -> str:
    """Get schema (PRAGMA table_info) and sample rows as string."""
    conn = sqlite3.connect(db_path)
    
    # Schema
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(data)")
    schema = cursor.fetchall()
    schema_str = "\n".join([f"{col[1]}: {col[2]}" for col in schema])
    
    # Sample
    df_sample = pd.read_sql_query("SELECT * FROM data LIMIT ?", conn, params=(sample_size,))
    sample_str = df_sample.to_csv(index=False)
    
    conn.close()
    return f"Schema (Column: Type):\n{schema_str}\n\nSample Data (first {sample_size} rows):\n{sample_str}"

def execute_sql(db_path: str, sql: str) -> pd.DataFrame:
    """Execute SQL and return DF."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def _get_dataset_preview(db_path: str) -> str:
    """Helper function to get a preview of the dataset (up to 50 rows and column names) from DB."""
    try:
        if not os.path.exists(db_path):
            return f"Error: Database '{db_path}' does not exist."

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        df = pd.read_sql_query("SELECT * FROM data LIMIT 50", conn)
        preview_rows = df.to_csv(index=False)
        conn.close()
        return f"Columns: {', '.join(columns)}\n\nFirst 50 rows:\n{preview_rows}"
    except Exception as e:
        return f"Error loading dataset preview: {str(e)}"

async def get_data_summary(db_path: str, query: str = "", conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """Provides a concise summary of the dataset in professional raw markdown format, tailored to the user's query if provided."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:

        # Get schema and sample data (20 rows)
        schema_sample = get_schema_and_sample(db_path, sample_size=20)
        
        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

        # Step 1: Generate SQL for summary
        sql_prompt = f"""
        Generate SQL queries to get summary statistics from the dataset.
        Use aggregations like COUNT, COUNT(DISTINCT), MIN, MAX, SUM for relevant columns.
        Return queries separated by ';'. Use the table 'data'. Do not add any prefixes or explanations.
        
        Schema and Sample (first 20 rows):
        {schema_sample}
        
        User Query: {query}
        
        Return ONLY SQL queries separated by ';'.
        """
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_queries = [clean_sql_query(q.strip()) for q in sql_response.split(';') if q.strip()]
        
        # Step 2: Execute SQL queries
        summary_data = ""
        for sql in sql_queries:
            try:
                logger.info(f"[GET_DATA_SUMMARY] Original SQL: {sql}")
                fixed_sql = force_data_table(sql)
                if sql != fixed_sql:
                     logger.info(f"[GET_DATA_SUMMARY] Sanitized SQL: {sql} -> {fixed_sql}")
                logger.info(f"[GET_DATA_SUMMARY] Executing SQL: {fixed_sql}")
                df = execute_sql(db_path, fixed_sql)
                result_csv = df.to_csv(index=False)
                logger.info(f"[GET_DATA_SUMMARY] SQL Result:\n{result_csv}")
                summary_data += f"SQL: {fixed_sql}\nResults:\n{result_csv}\n\n"
            except Exception as e:
                logger.warning(f"[GET_DATA_SUMMARY] SQL Error: {str(e)}")
                summary_data += f"Error in SQL '{sql}': {str(e)}\n\n"

        # Step 3: Format response with LLM
        prompt = f"""        You are a Senior Financial Analyst. Produce a brief, well-structured financial summary in professional RAW MARKDOWN ONLY. Use bold labels and bullets.
        Requirements:
        - Keep it concise (max 6 bullets total); no long paragraphs
        - Use bold labels (e.g., **Revenue Overview**, **Top Expenses**, **Budget Status**)
                - No backticks, no extra wrappers, no 'Final Answer' headings
   - Return ONLY markdown; no prose outside the markdown
        - Analyze the user query and conversation history to tailor the response
        - Ensure consistency with previous responses in the conversation history
        - Analyse the user query and generate the response according to that, make the response always brief
        - Wrap the response with an introductory message for example like this 'Here's the response I got for you:' and a follow-up 'Let me know if you need further assistance or have another query!' or followup query based on the response - dont use the exact message you can generate this message on your own
        - **CRITICAL**: Do NOT output Python code or scripts. If the user asked for a program, provide the *result* of the analysis instead.
        - Allow markdown bolding/italics for emphasis
        - Return ONLY markdown, no prose outside the markdown
        - Analyze the user query through a financial lens (ROI, Margin, Growth)
        - Wrap the response with a professional introductory message and a financial follow-up.
        - **CRITICAL**: Do NOT output Python code. Provide financial results instead.

        Chat History:
        {chat_history}

        User Query: {query}
        
        Schema and Sample Data (first 20 rows):
        {schema_sample}
        
        SQL Query Results:
        {summary_data}
        """
        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        return response_obj.content, total_usage
    except Exception as e:
        logger.error(f"[GET_DATA_SUMMARY] Error: {str(e)}")
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return "I'm currently experiencing a high volume of requests. Please try again in a moment, and I'll be ready to assist you!", total_usage
        return f"I encountered an issue while summarizing the data. Please try again, or let me know if you have another question.", total_usage

async def detect_anomalies(db_path: str, query: str = "", conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """Analyzes every row of the dataset to identify anomalies, providing the response in professional raw markdown format."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:


        dataset_preview = _get_dataset_preview(db_path)
        if dataset_preview.startswith("Error"):
            return f"- {dataset_preview}", total_usage

        # For anomalies, generate SQL to detect issues (e.g., NULL counts, outliers via stats)
        schema_sample = get_schema_and_sample(db_path, sample_size=20)
        
        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

        # First, let LLM generate SQL for anomaly detection
        sql_prompt = f"""
        Based on the schema and sample, generate SQL queries to detect anomalies like NULLs, duplicates, outliers (use AVG/STDEV for numeric).
        Return a list of SQL queries separated by ';'. Use the table 'data'. Do not add any prefixes or explanations.
        Schema and Sample:
        {schema_sample}
        User Query: {query}
        """
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_queries = [clean_sql_query(q.strip()) for q in sql_response.split(';') if q.strip()]

        anomalies_data = ""
        for sql in sql_queries:
            try:
                logger.info(f"[DETECT_ANOMALIES] Original SQL: {sql}")
                fixed_sql = force_data_table(sql)
                if sql != fixed_sql:
                     logger.info(f"[DETECT_ANOMALIES] Sanitized SQL: {sql} -> {fixed_sql}")
                logger.info(f"[DETECT_ANOMALIES] Executing SQL: {fixed_sql}")
                df = execute_sql(db_path, fixed_sql)
                result_csv = df.to_csv(index=False)
                logger.info(f"[DETECT_ANOMALIES] SQL Result:\n{result_csv}")
                anomalies_data += f"SQL: {fixed_sql}\nResults:\n{result_csv}\n\n"
            except Exception as e:
                logger.warning(f"[DETECT_ANOMALIES] SQL Error: {str(e)}")
                anomalies_data += f"Error in SQL '{sql}': {str(e)}\n\n"

        prompt = f"""
        You are a Financial Auditor. Produce a brief, structured financial anomaly/audit report in RAW MARKDOWN ONLY. Use bold labels and bullets.
        Requirements:
        - Maximum 7 bullets total. Identify financial inconsistencies, suspicious variances, or data gaps.
        - Return ONLY markdown with bold highlights.
        - Analyze the user query and conversation history to tailor the response
        - Ensure consistency with previous responses in the conversation history
        - Wrap the response with an introductory message for example like this  'Here's the response I got for you:' and a follow-up 'Let me know if you need further assistance or have another query!' or followup query based on the response - Note : dont use the exact message you can generate this message on your own
        - **CRITICAL**: Do NOT output Python code or scripts. If the user asked for a program, provide the *result* of the analysis instead.

        Structure to follow:
        **Data Quality & Anomalies**  
        - **Findings**:  
          - [Column]: [issue summary]
          - If none: "No anomalies detected."
        - **Recommendations**:  
          - [Brief, actionable fix]

        analyse the full available dataset , dont miss any outliers , you should identify all type of anomolies like missing datas, irrrelevant data, logically incorrect etc
        analyse each abd every row connnect with the other column of data etc

        Chat History:
        {chat_history}

        User Query: {query}

        Dataset Preview (first 50 rows for context): {dataset_preview}
        Anomaly Query Results: {anomalies_data}
        """
        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        return response_obj.content, total_usage
    except Exception as e:
        logger.error(f"[DETECT_ANOMALIES] Error: {str(e)}")
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return "I'm currently experiencing a high volume of requests. Please try again in a moment, and I'll be ready to assist you!", total_usage
        return f"I encountered an issue while auditing the data. Please try again shortly.", total_usage

async def generate_insights(db_path: str, query: str = "", conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """Generates high-level insights from the dataset, providing the response in professional raw markdown format."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:

        dataset_preview = _get_dataset_preview(db_path)
        if dataset_preview.startswith("Error"):
            return f"- {dataset_preview}", total_usage

        # Generate SQL for insights (aggregates)
        schema_sample = get_schema_and_sample(db_path, sample_size=20)
        available_columns = get_columns_list(db_path)
        dataset_preview = _get_dataset_preview(db_path)
        
        sql_prompt = f"""
        Generate SQL queries for key insights (aggregations, grouping, rankings, etc.).
        
        CRITICAL RULES:
        1. Use ONLY SELECT statements
        2. Use table name EXACTLY 'data' for all table references in WHERE/JOIN clauses
        3. Use the correct column names from the schema (case-sensitive)
        4. IMPORTANT: When using CTEs (WITH ... AS), the final SELECT must query FROM the CTE name, NOT from 'data'
           Example: WITH RankedEmployees AS (SELECT ... FROM data) SELECT ... FROM RankedEmployees WHERE rn = 1
        5. For window functions (ROW_NUMBER, RANK, etc), SQLite may not support all - use subqueries or GROUP BY alternatives if needed
        6. Return queries separated by ';'.
        7. Do not add any prefixes or explanations
        
        DATASET INFORMATION:
        Available Columns: {', '.join(available_columns)}
        
        Schema and Sample (first 20 rows):
        {schema_sample}
        
        Dataset Preview (first 50 rows for reference):
        {dataset_preview}
        
        User Query: {query}
        
        Return ONLY the SQL queries, nothing else. No explanations, no markdown.
        """
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_queries = [clean_sql_query(q.strip()) for q in sql_response.split(';') if q.strip()]

        insights_data = ""
        for sql in sql_queries:
            try:
                logger.info(f"[GENERATE_INSIGHTS] Original SQL: {sql}")
                fixed_sql = force_data_table(sql)
                if sql != fixed_sql:
                    logger.info(f"[GENERATE_INSIGHTS] Sanitized SQL: {sql} -> {fixed_sql}")
                logger.info(f"[GENERATE_INSIGHTS] Executing SQL: {fixed_sql}")
                df = execute_sql(db_path, fixed_sql)
                result_csv = df.to_csv(index=False)
                logger.info(f"[GENERATE_INSIGHTS] SQL Result:\n{result_csv}")
                insights_data += f"SQL: {fixed_sql}\nResults:\n{result_csv}\n\n"
            except Exception as e:
                logger.warning(f"[GENERATE_INSIGHTS] SQL Error: {str(e)}")

        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

        prompt = f"""        You are a Senior Strategic Financial Analyst. Produce brief, high-signal financial insights in RAW MARKDOWN ONLY.
        Requirements:
        - Max 6 bullets. Focus on profitability, cost-saving, revenue growth, and risk.
        - Use bold category labels (e.g., **Profitability Analysis**, **Cash Flow Trends**)
        - Return ONLY markdown with bold emphasis.
        - Analyse the each and every row and identify the key insights as for the decision making
        - Wrap the response with an introductory message for example like this  'Here's the response I got for you:' and a follow-up 'Let me know if you need further assistance or have another query!' or followup query based on the response - Note: dont use the exact message you can generate this message on your own
        - **CRITICAL**: Do NOT output Python code or scripts. If the user asked for a program, provide the *result* of the analysis instead.

        Chat History:
        {chat_history}

        User Query: {query}

        Dataset Preview (first 50 rows for context): {dataset_preview}
        Insights Data: {insights_data}
        """
        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        return response_obj.content, total_usage
    except Exception as e:
        logger.error(f"[GENERATE_INSIGHTS] Error: {str(e)}")
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return "I'm currently experiencing a high volume of requests. Please try again in a moment, and I'll be ready to assist you!", total_usage
        return f"I encountered an issue while generating insights. Could you try rephrasing your request?", total_usage

async def query_dataset(db_path: str, query: str, conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """Answers specific queries about the dataset by generating and executing SQL, with AI-powered column mismatch detection."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:

        if not os.path.exists(db_path):
            return f"- Error: Database '{db_path}' not found.\n- Query: {query}", total_usage

        schema_sample = get_schema_and_sample(db_path, sample_size=20)
        dataset_preview = _get_dataset_preview(db_path)  # First 50 rows
        available_columns = get_columns_list(db_path)

        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

            logger.info("[QUERY_DATASET] Running column mismatch validation with LLM")
        # Step 1: AI-powered validation - Let LLM analyze query against dataset headers and sample data
        validation_prompt = f"""
        You are an intelligent data analyst assistant. Analyze the user's query for potential column name mismatches by carefully studying the available dataset.
        
        AVAILABLE DATASET INFORMATION:
        
        Column Headers: {', '.join(available_columns)}
        
        Dataset Preview (first 50 rows for context):
        {dataset_preview}
        
        User Query: {query}
        
        Your task:
        1. Carefully analyze the user's query to identify any column names or field references they mentioned.
        2. Match each mentioned field against the available columns by:
           - Checking exact matches (case-insensitive)
           - Analyzing the column names semantically (e.g., "role" might match "job_title", "position")
           - Looking at the sample data to understand what each column contains
           - Considering common synonyms and variations
        3. For each potential mismatch found, identify the best matching columns from the dataset.
        4. Respond ONLY with a valid JSON object in this exact format:
           {{"has_mismatch": false}}
           OR
           {{"has_mismatch": true, "mismatches": [{{"user_mentioned": "field_name", "best_matches": ["column1", "column2"]}}]}}
        
        IMPORTANT: Return ONLY valid JSON, nothing else. No extra text, no explanations.
        """
        
        validation_response_obj = await llm.ainvoke(validation_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(validation_response_obj))
        validation_response = validation_response_obj.content.strip()
        try:
            validation_result = json.loads(validation_response)
            logger.info(
                "[QUERY_DATASET] Parsed validation JSON",
                extra={"validation_result": validation_result},
            )
        except json.JSONDecodeError:
            logger.warning(f"[QUERY_DATASET] Failed to parse validation JSON, retrying...")
            # Retry with simpler prompt
            validation_result = {"has_mismatch": False}
        
        # If column mismatches detected, ask user for clarification using AI-generated message
        if validation_result.get("has_mismatch", False):
            mismatches = validation_result.get("mismatches", [])
            logger.info(
                "[QUERY_DATASET] Column mismatches detected",
                extra={"mismatches": mismatches},
            )

            if mismatches:
                # Use AI to generate a natural, dynamic clarification message
                clarification_prompt = f"""
                You are a helpful AI data analyst. A user has asked a query, but we've detected potential column name mismatches.
                
                User Query: {query}
                
                Mismatches found:
                {json.dumps(mismatches, indent=2)}
                
                Dataset Column Headers: {', '.join(available_columns)}
                
                Dataset Preview (first 50 rows for reference):
                {dataset_preview}
                
                Your task:
                1. Generate a natural, friendly clarification message (2-3 sentences max) that:
                   - Politely explains that we couldn't find exact matches for the columns they mentioned
                   - Suggests the best matching columns based on the mismatches
                   - Asks them to clarify which column they meant or provide more details
                2. Use the sample data context to make informed suggestions
                3. Sound conversational, helpful, and data-aware (reference actual column names and what they contain)
                4. Return ONLY the clarification message text, no JSON, no extra formatting
                dont make the clarification message too vague , ask with some options or choice
                
                Example tone: "I don't see a 'role' column, but based on your query and the dataset, I found 'job_title' and 'position' which might be what you're looking for. Which one would you like to use?"
                """
                
                clarification_response_obj = await llm.ainvoke(clarification_prompt)
                _accumulate_usage(total_usage, _extract_token_usage(clarification_response_obj))
                clarification_response = clarification_response_obj.content.strip()
                logger.info(f"[QUERY_DATASET] Column mismatch detected, generated clarification")
                return clarification_response, total_usage
            else:
                logger.info("[QUERY_DATASET] No column mismatches detected by validation step")

        # Step 2: Generate SQL (with full context including columns and sample data)
        sql_prompt = f"""
        You are a SQL expert. Generate a valid SQL query for the table 'data' to answer the user's query.
        
        CRITICAL RULES:
        - Use ONLY SELECT statements
        - Use ONLY these exact column names that exist in the database: {', '.join(available_columns)}
        - Do NOT invent or assume column names
        - All column names are lowercase with underscores
        
        Dataset Overview:
        {schema_sample}
        
        Dataset Sample (first 50 rows):
        {dataset_preview}

        Chat History:
        {chat_history}

        Query: {query}

        Return ONLY the SQL query, nothing else. Do not add any prefix like 'SQL:' or explanations. No markdown, no code blocks. avoid amiable 
        """
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_query = clean_sql_query(sql_response)
        logger.info(
            "[QUERY_DATASET] Cleaned SQL query",
            extra={"sql_query": sql_query},
        )
        print(f"Generated SQL Query: {sql_query}")

        # Step 3: Execute SQL
        fixed_sql_query = force_data_table(sql_query)
        logger.info(
            "[QUERY_DATASET] Fixed SQL query with 'data' table applied",
            extra={"fixed_sql_query": fixed_sql_query},
        )
        try:
            result_df = execute_sql(db_path, fixed_sql_query)
            results_str = result_df.to_csv(index=False)
            logger.debug(
                "[QUERY_DATASET] SQL results converted to CSV",
                extra={"results_preview": results_str[:500]},
            )
            print(f"SQL Execution Results:\n{results_str}")
        except Exception as exec_e:
            logger.error(f"[QUERY_DATASET] SQL Execution Error: {str(exec_e)}")
            # Use AI to generate helpful error recovery message
            error_recovery_prompt = f"""
            The SQL query failed to execute. Generate a natural, helpful error message that:
            1. Acknowledges the issue
            2. References the available columns
            3. Asks the user to clarify their intent
            
            SQL Error: {str(exec_e)}
            Available Columns: {', '.join(available_columns)}
            User Query: {query}
            
            Return ONLY the error recovery message (1-2 sentences max), conversational and helpful tone.
            """
            error_msg_obj = await llm.ainvoke(error_recovery_prompt)
            _accumulate_usage(total_usage, _extract_token_usage(error_msg_obj))
            error_msg = error_msg_obj.content.strip()
            return error_msg, total_usage

        # Step 4: Let LLM format the response
        format_prompt = f"""
         You are the **BP Store Manager AI Copilot**. Provide a concise, structured answer in RAW MARKDOWN ONLY based on the SQL results and the user's query.

        Requirements:
        - Use bold labels and bullets. For single specific records (like a specific invoice or transaction), provide a comprehensive breakdown including seller/company info, customer info, items, and taxes.
        - Do not arbitrarily limit the number of bullets for specific record lookups; ensure all relevant details from the data are presented.
        - Include the computed value(s) explicitly with units/currency
        - If aggregation is needed, state the method briefly (sum/avg/group-by)
        - Analyze the conversation history to ensure consistency with previous responses
        - Return ONLY markdown  
        - Analyze each and every row of data, do some logical thinking and reasoning, and check whether the response is correct
        - Try to identify the user intent; if you have a doubt, ask for clarification from the user about the query
        - Check the previous answer to avoid any controversy in the response
        - **CRITICAL**: Do NOT output Python code or scripts. If the user asked for a program, provide the *result* of the analysis instead.

        Chat History:
        {chat_history}

        SQL Query Executed: {fixed_sql_query}
        Results (as CSV): {results_str}
        User Query: {query}
        """
        response_obj = await llm.ainvoke(format_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        logger.info(
            "[QUERY_DATASET] Successfully generated final markdown answer",
            extra={"answer_preview": response_obj.content[:500]},
        )
        return response_obj.content, total_usage
        
    except Exception as e:
        logger.error(f"[QUERY_DATASET] Error: {str(e)}")
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return "I'm currently experiencing a high volume of requests. Please try again in a moment, and I'll be ready to assist you!", total_usage
        return f"I encountered an error while querying the dataset. Please try again.", total_usage

async def generate_detailed_report(db_path: str, query: str = "", conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """Generates a detailed report about the dataset, providing the response in professional raw markdown format."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:

        dataset_preview = _get_dataset_preview(db_path)
        if dataset_preview.startswith("Error"):
            return f"- {dataset_preview}", total_usage

        # Get stats via SQL
        conn = sqlite3.connect(db_path)
        row_count = pd.read_sql_query("SELECT COUNT(*) as count FROM data", conn).iloc[0]['count']
        columns = pd.read_sql_query("PRAGMA table_info(data)", conn)['name'].tolist()
        conn.close()

        # Generate SQL for report sections
        schema_sample = get_schema_and_sample(db_path, sample_size=20)
        sql_prompt = f"""
        Generate SQL queries for a summary report: total rows, unique counts per column, sums/avgs for numerics, top groups.
        IMPORTANT RULES:
        - Use ONLY SELECT statements.
        - Use the table name exactly 'data' (do not invent names like 'orders' or placeholders like 'your_table_name').
        - Separate multiple queries by ';'.
        - Do not add any prefixes or explanations.
        
        Schema: {schema_sample}
        """
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_queries = [clean_sql_query(q.strip()) for q in sql_response.split(';') if q.strip()]

        report_data = f"Row count: {row_count}, Columns: {', '.join(columns)}\n"
        for sql in sql_queries:
            try:
                logger.info(f"[GENERATE_DETAILED_REPORT] Original SQL: {sql}")
                fixed_sql = force_data_table(sql)
                if sql != fixed_sql:
                    logger.info(f"[GENERATE_DETAILED_REPORT] Sanitized SQL: {sql} -> {fixed_sql}")
                logger.info(f"[GENERATE_DETAILED_REPORT] Executing SQL: {fixed_sql}")
                df = execute_sql(db_path, fixed_sql)
                result_csv = df.to_csv(index=False)
                logger.info(f"[GENERATE_DETAILED_REPORT] SQL Result:\n{result_csv}")
                report_data += f"SQL: {fixed_sql}\n{result_csv}\n\n"
            except Exception as e:
                logger.warning(f"[GENERATE_DETAILED_REPORT] SQL Error: {str(e)}")
                pass

        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

        prompt = f"""
        You are a CFO's Chief of Staff. Generate a concise, professional Financial Report in RAW MARKDOWN ONLY with bold labels and headers.
        Focus on:
        1. Financial Performance (Revenue, GP, EBITDA if applicable)
        2. Operational Efficiency
        3. Risk & Compliance (Audit findings)
        4. Forward-looking recommendations.
        Return ONLY markdown.
        Constraints:
        - Each section max 3-5 bullets; 1 line per bullet
        - Use bold labels for metrics; include currency where relevant (₹)
        - Analyze the conversation history to ensure consistency with previous responses
        - Return ONLY markdown , while telling the anomolies are inconsistency dont mention the rows number , just tell it in common, for example some rows of data contains nulllv
        - **CRITICAL**: Do NOT output Python code or scripts. If the user asked for a program, provide the *result* of the analysis instead.

        For example : Note this is only for the example , take formatting structure alone not the content 
        Structure to follow:
        ##Dataset Overview  
        - [1-2 lines on purpose and key attributes]

        **Summary Statistics**  
        - **Total Orders**: [n]
        - **Unique Customers**: [n]
        - **Unique Products**: [n]
        - **Total Quantity Ordered**: [n]
        - **Total Revenue Generated**: ₹[amount]
        - **Payment Methods**: [comma-separated]
        - **Cities Covered**: [comma-separated]

        **Data Quality & Anomalies**  
        - **Findings**: [up to 2-3 bullets specifying column, row index, issue]
        - **Recommendations**: [1-2 bullets]

        **Product Highlights**  
        - **Top Revenue Products**: [up to 3 items]
        - **Most Ordered**: [product]

        **City-Wise Insights**  
        - [up to 3 bullets of city and revenue]
        - **Top City**: [city]
        - **Lowest City**: [city]

        **Key Insights**  
        - [2-4 bullets of actionable insights]

        **Conclusion**  
        - [1 line summary and next action]

        Chat History:
        {chat_history}

        User Query: {query}
        Dataset Preview (first 50 rows for context): {dataset_preview}
        Report Data: {report_data}
        """
        # Generate the report in MARKDOWN (existing)
        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))

        markdown_report = response_obj.content if hasattr(response_obj, "content") else str(response_obj).strip()

        # --- NEW: Convert Markdown to HTML using the LLM ---
        # We ask the model to produce a clean, mobile-friendly HTML document fragment
        # that uses minimal styling. We instruct it to output only HTML (no wrappers).
        convert_prompt = f"""
You are a professional HTML report generator. Convert the following RAW MARKDOWN REPORT into a clean, structured, and modern HTML fragment using the EXACT styling, measurements, and structure defined below.

### MANDATORY RULES (NON-NEGOTIABLE)
- Output ONLY the final HTML fragment. No explanations, no comments, no markdown fences, no extra text.
- Use EXACTLY the provided <style> block (copy-paste it verbatim — do not modify anything).
- Structure MUST follow this exact pattern:
  - One root <div class="ai-report">
  - One <h1 class="report-title"> with the report title (use "Analysis Report" if none specified)
  - Every major section MUST be wrapped in <section class="ai-card">
  - Every section title MUST use <h3 class="section-heading">
  - Never use <h1> or <h2> — only <h3> and <h4> allowed
  - Lists: use <ul> with custom arrow bullets (via CSS)
  - Numbers & metrics: wrap in <span class="highlight-number">
  - Important text/phrases: wrap in <span class="highlight-text">
  - Technical terms & code: wrap in <span class="highlight-term"><code>term</code></span>
- Always use semantic HTML and proper spacing.

### EXACT STYLE TO USE (COPY THIS VERBATIM INSIDE THE HTML)

<div class="ai-report">
  <style>
    /* --- Page Layout & Typography --- */
    .ai-report {{
      font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
      color: #444;
      background-color: #f8fafc;
      max-width: 900px;
      margin: 2rem auto;
      padding: 1rem;
      line-height: 1.8;
      box-sizing: border-box;
      border: 1px solid rgba(37, 99, 235, 0.25);
      border-radius: 12px;
      box-shadow: 0 0 20px rgba(37, 99, 235, 0.08);
    }}
    .report-title {{
      font-size: 2.2rem;
      font-weight: 700;
      color: #0f2d5c;
      text-transform: uppercase;
      letter-spacing: 2px;
      border-bottom: 2px solid #e2e8f0;
      text-align: center;
      margin: 0 0 1.5rem 0;
      padding-bottom: 1rem;
    }}
    .ai-card {{
      background-color: #ffffff;
      border-radius: 8px;
      border-top: 4px solid #2563eb;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
      padding: 1rem;
      margin-bottom: 1rem;
      transition: transform 0.2s ease;
    }}
    .ai-card:hover {{transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }}
    .section-heading {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #1e3a8a;
      margin: 0 0 1.25rem 0;
      display: flex;
      align-items: center;
    }}
    .section-heading::before {{
      content: '';
      display: block;
      width: 8px;
      height: 24px;
      background-color: #3b82f6;
      margin-right: 12px;
      border-radius: 2px;
    }}
    p {{ margin-bottom: 1rem; color: #4b5563; font-size: 1.05rem; }}
    ul {{list-style: none; padding: 0; margin: 0; }}
    li {{
      margin-bottom: 0.8rem;
      padding-left: 1.5rem;
      position: relative;
      color: #374151;
    }}
    li::before {{
      content: "›";
      font-size: 1.5rem;
      line-height: 1rem;
      color: #2563eb;
      position: absolute;
      left: 0;
      top: 5px;
      font-weight: bold;
    }}
    li strong {{ color: #111827; font-weight: 700; margin-right: 4px; }}
    .highlight-number {{
      color: #2563eb;
      font-weight: 700;
      font-family: 'Segoe UI', monospace;
      font-size: 1.05em;
    }}
    .highlight-text {{
      color: #1e3a8a;
      font-weight: 700;
    }}
    .highlight-term {{
      color: #0891b2;
      font-weight: 600;
    }}
    .highlight-term code {{
      font-family: "Consolas", "Monaco", monospace;
      font-size: 0.95em;
      color: #0891b2;
      background: none;
      padding: 0;
    }}
  </style>

  <h1 class="report-title">Analysis Report</h1>

  <!-- CONVERT MARKDOWN SECTIONS INTO ai-card BLOCKS BELOW THIS LINE -->

  {{PLACEHOLDER_FOR_CONVERTED_CARDS}}

</div>

### INSTRUCTIONS FOR CONVERSION
- Convert each top-level heading (## Title) → <section class="ai-card"><h3 class="section-heading">Title</h3>...
- Convert bullet points → <ul><li>...
- Bold text **like this** → <strong> or appropriate highlight class
- Inline code `column_name` → <span class="highlight-term"><code>column_name</code></span>
- Numbers that are metrics → <span class="highlight-number">123</span>
- Important phrases → <span class="highlight-text">Important</span>

### INPUT MARKDOWN
---START_MARKDOWN---
{markdown_report}
---END_MARKDOWN---

Return ONLY the complete HTML with the exact style above and properly converted content. No deviations allowed.
        """


        try:
            html_response_obj = await llm.ainvoke(convert_prompt)
            _accumulate_usage(total_usage, _extract_token_usage(html_response_obj))
            html_report = html_response_obj.content.strip() if hasattr(html_response_obj, "content") else str(html_response_obj).strip()
        except Exception as e:
            logger.warning(f"[GENERATE_DETAILED_REPORT] HTML conversion failed: {str(e)}")
            # fallback: very basic markdown->html conversion for safety
            safe_html = markdown_report.replace("\n\n", "</p><p>").replace("\n", "<br/>")
            html_report = f"<div class='prose prose-sm'>{safe_html}</div>"

        # Return both markdown and html as a JSON string so callers can choose
        return json.dumps({
            "markdown": markdown_report,
            "html": html_report
        }), total_usage

    except Exception as e:
        logger.error(f"[GENERATE_DETAILED_REPORT] Error: {str(e)}")
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower():
            return json.dumps({
                "markdown": "I'm currently experiencing a high volume of requests. Please try again in a moment.",
                "html": "<div class='ai-report'><p>I'm currently experiencing a high volume of requests. Please try again in a moment.</p></div>"
            }), total_usage
        return f"I encountered an issue while generating the report. Please try again.", total_usage

async def generate_visualization(db_path: str, query: str, conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """
    Generates a visualization dynamically based on user query, conversation history, and dataset using Gemini + PythonREPLTool.
    Returns JSON: {chart_type, chart_code, image, explanation}
    Uses SQL to fetch relevant data subset.
    """
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        schema_sample = get_schema_and_sample(db_path, sample_size=20)

        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])

        # Step 1: Generate SQL to fetch viz data
        sql_prompt = f"""
        Generate SQL to fetch data for visualization.
        IMPORTANT RULES:
        - Use ONLY SELECT statements.
        - Use the table name exactly 'data' (do not invent or use placeholders like 'your_table_name').
        - Do not add any prefixes or explanations.
        
        Schema: {schema_sample}
        Chat History: {chat_history}
        User Query: {query}
        
        Return ONLY the SQL query, nothing else. Do not add any prefix like 'SQL:' or explanations. No markdown, no code blocks.
        """
        logger.info(f"[GENERATE_VISUALIZATION] Generating SQL for visualization")
        sql_response_obj = await llm.ainvoke(sql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(sql_response_obj))
        sql_response = sql_response_obj.content.strip()
        sql_query = clean_sql_query(sql_response)
        logger.info(f"[GENERATE_VISUALIZATION] LLM SQL (raw): {sql_query}")

        # Sanitize table name to ensure 'data' is used
        original_sql_query = sql_query
        sql_query = re.sub(r"\byour_table_name\b", "data", sql_query, flags=re.IGNORECASE)
        if original_sql_query != sql_query:
           logger.info(f"[GENERATE_VISUALIZATION] Sanitized SQL: {original_sql_query} -> {sql_query}")

        # Ensure a LIMIT 1000 exists to avoid huge results (if none present)
        if re.search(r"\blimit\b", sql_query, flags=re.IGNORECASE) is None:
            sql_query = sql_query.rstrip("; ") + " LIMIT 1000"
            logger.info(f"[GENERATE_VISUALIZATION] Added LIMIT: {sql_query}")

        # Execute SQL to get viz_df
        logger.info(f"[GENERATE_VISUALIZATION] Executing SQL: {sql_query}")
        viz_df = execute_sql(db_path, sql_query)
        logger.info(f"[GENERATE_VISUALIZATION] SQL returned {len(viz_df)} rows")
        viz_info = f"Columns: {', '.join(viz_df.columns.tolist())}\nFirst 5 rows:\n{viz_df.head(5).to_string(index=False)}"

        # Step 2: Generate chart code using viz_df
        prompt = f"""
        You are a senior Python data visualization engineer. Produce only valid, runnable Python code (no markdown, no comments, no explanation) that creates a high-quality visualization from a pandas DataFrame named `df`. The environment is a headless server — DO NOT use or import any GUI or tkinter-related modules (e.g., tkinter, PIL.ImageTk, TkAgg, plt.show(), plt.ion(), or any GUI backends). Use a non-GUI backend for matplotlib.

    Dataset Overview (from SQL):
    {viz_info}

    Chat History:
    {chat_history}

    User Query:
    {query}

    Requirements (strict):
    1. At the very start of the code ensure matplotlib uses a non-GUI backend: set `matplotlib.use("Agg")` before importing `matplotlib.pyplot`.
    2. Only import standard, widely-available plotting libraries: `matplotlib` and/or `seaborn` and necessary pandas/numpy utilities. Do NOT import tkinter, PIL.ImageTk, or any GUI-specific backends.
    3. The code MUST use the variable `df` that is already provided. Do not re-read files or create I/O except saving the figure object to a variable.
    4. Create the visualization and assign the matplotlib Figure object to a variable named exactly `fig`.
    5. Do NOT call `plt.show()`, `plt.ion()`, or any function that requires a GUI event loop.
    6. Save nothing to disk. Do not call `fig.savefig(...)`. (The system will handle saving/encoding.)
    7. Automatically detect the best chart type for the requested query (bar, line, scatter, pie, hist, box, heatmap, etc.). If multiple series are implied, choose an appropriate multi-series plot.
    8. Use only from these listed attributes in the program: chart_type, x, y, values, labels, title, xlabel, ylabel, figsize, dpi, color, colors, alpha, linewidth, linestyle, marker, markersize, markerfacecolor, markeredgecolor, grid, legend, xticks, yticks, xticklabels, yticklabels, xlim, ylim, width, orientation, edgecolor, bar_labels, s, c, cmap, edgecolors, linewidths, vmin, vmax, markerstyle, autopct, explode, shadow, startangle, radius, pctdistance, labeldistance, wedgeprops, textprops, loc, horizontalalignment, verticalalignment
    At the very start of the code, you MUST import matplotlib, then immediately call:
   import matplotlib
   matplotlib.use("Agg")

   Only after this may you import matplotlib.pyplot as plt.
    Make code robust: validate column presence, handle datetime parsing when necessary, handle missing values, and fall back gracefully if the chosen columns are not present (e.g., try alternatives or produce a simple table plot).
    9. Use modern, clean aesthetics (matplotlib style or seaborn style), appropriate figure size, tight layout, readable labels, rotated x-ticks when needed, legend, grid, and color palette. Keep visuals uncluttered.
    10. Ensure numeric aggregation when needed (e.g., groupby + sum/mean/count) based on the query intent.
    11. Maintain consistency with earlier visualizations in `{chat_history}` by using similar color palette/styles if prior visuals are present.
    12. Do not use the substring "ha" anywhere in the generated code (avoid this token).
    dont use the attributes that are not available ,dont use the attributes like set_layout_tight" use the common and mostly used attributes only, dont make the code more complex with adding more features , just a simple visualisation with data, so use the repeated attributes only
    13. Ensure the produced code is syntactically correct with no undefined names; use defensive programming (e.g., `if "<col>" in df.columns:`) and clear variable naming.
    14. CRITICAL - Use ONLY valid matplotlib parameters: For pie charts use `startangle` NOT `st_angle`, use `autopct` NOT `autopct_format`, use `labels` NOT `label`. Verify all parameters are valid before using them.
    15. Return ONLY Python code. No extra text, no comments, no markdown, and no explanatory output.
    16. **Suggestion Handling**: If the user asks for "all possible charts" or "suggest a chart", choose the SINGLE most informative visualization for the dataset (e.g., a correlation heatmap or a bar chart of the most cardinal categorical column). Do NOT try to generate multiple charts in one go.
    17. **Robust X-Axis Handling**: NEVER pass `df.index` directly to the `x` parameter of `df.plot()`. If you want to use the index as categories, either omit the `x` parameter entirely or reset the index first (`df.reset_index()`) and use the new column name.
    18. **Financial Totals Handling**: If the dataset has only 1 row (common for total/sum results), consider a horizontal bar chart or a pie chart to compare the columns, or transpose the data so columns become categories.
    Produce final code that when executed in a headless FastAPI worker environment will set `fig` to the figure object for the requested visualization and will not raise GUI-related errors.
         Return ONLY valid Python code. No markdown, no explanation, no comments. avoid amiable , dont give any marks like "```python" or "```"
 """

        response_obj = await llm.ainvoke(prompt)
        _accumulate_usage(total_usage, _extract_token_usage(response_obj))
        chart_code = response_obj.content.strip()
        # Clean markdown fences if present
        chart_code = chart_code.replace("```python", "")
        chart_code = chart_code.replace("```", "")
        chart_code = chart_code.strip()
        
        print("Generated Chart Code:\n", chart_code)

        if not chart_code or "import" not in chart_code:
            return json.dumps({
                "chart_type": None,
                "chart_code": chart_code,
                "image": None,
                "explanation": "Error: No valid visualization code generated."
            }), total_usage

        # Step 3: Execute the visualization code
        python_tool = PythonREPLTool()
        local_env = {"df": viz_df, "plt": plt, "pd": pd}
        exec(chart_code, local_env)

        # Step 4: Save figure to image
        fig = local_env.get("fig", None)
        img_base64 = None

        if fig is None:
            raise ValueError("No figure object ('fig') was generated in code.")

        # Handle Plotly visualization
        # if "plotly" in chart_code.lower():
        #     import plotly.io as pio
        #     buf = io.BytesIO()
        #     # Convert Plotly fig to image
        #     img_bytes = fig.to_image(format="png")
        #     img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # # Handle Bokeh visualization
        # elif "bokeh" in chart_code.lower():
        #     from bokeh.io.export import get_screenshot_as_png
        #     from PIL import Image
        #     img = get_screenshot_as_png(fig)
        #     buf = io.BytesIO()
        #     img.save(buf, format="PNG")
        #     img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Default: assume Matplotlib or Seaborn
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)


        # Step 5: Short, 1–2 line explanation
        explain_prompt = f"""
        Provide a brief 1–2 line insight about the visualization generated from the dataset, focusing only on the data insights derived from the visualization. Do not mention user intent or phrases like 'the user wants'. Consider the conversation history to maintain context and consistency. Return ONLY the insight text.
        Chat History: {chat_history}
        Query: {query}
        Dataset Info: {viz_info}
        """
        explanation_obj = await llm.ainvoke(explain_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(explanation_obj))
        explanation = explanation_obj.content.strip()

        return json.dumps({
            "chart_type": "auto",
            "chart_code": chart_code,
            "image": img_base64,
            "explanation": explanation
        }), total_usage

    except Exception as e:
        logger.error(f"[GENERATE_VISUALIZATION] Error: {str(e)}")
        error_str = str(e)
        friendly_msg = f"Error generating visualization: {str(e)}"
        if "429" in error_str or "rate limit" in error_str.lower():
            friendly_msg = "I'm currently experiencing a high volume of requests. Please wait a moment and try again so I can generate your chart!"
            
        return json.dumps({
            "chart_type": None,
            "chart_code": None,
            "image": None,
            "explanation": friendly_msg
        }), total_usage

async def query_salesforce_customer(query: str, conversation: list = [], context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """
    Queries Salesforce for customer details based on the user's natural language request.
    It fetches fields metadata dynamically to construct a correct SOQL query via LLM.
    """
    from app.external_services.salesforce import get_object_fields, query_salesforce
    
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    try:
        object_name = os.getenv("SF_CUSTOMER_OBJECT_NAME", "Customer_Details__c")
        
        # 1. Fetch Salesforce object fields dynamically
        fields = get_object_fields(object_name)
        if not fields:
            return f"Error: Could not retrieve metadata description for Salesforce object '{object_name}'. Please verify that the object exists in Salesforce and credentials are correct.", total_usage
            
        fields_summary = "\n".join([f"- {f['name']} ({f['type']}): {f['label']}" for f in fields])
        
        # Prepare conversation context
        chat_history = ""
        if context_summary:
            chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
        if conversation:
            chat_history += "".join([f"User: {item['user']}\nAssistant: {item['ai']}\n" for item in conversation])
            
        # 2. Ask LLM to generate SOQL
        soql_prompt = f"""
        You are a Salesforce SOQL query expert.
        Generate a valid Salesforce SOQL query for the custom object '{object_name}' to answer the user's query.
        
        CRITICAL RULES:
        - Use ONLY SELECT statements.
        - Do NOT use 'SELECT *'. List only the API field names to retrieve.
        - Use ONLY fields that exist in the schema below:
        {fields_summary}
        - Do NOT invent field names.
        - String literals in the WHERE clause must be enclosed in single quotes.
        - Do NOT include markdown styling or headers. Return ONLY the raw SOQL string.
        - If matching names, use the LIKE operator with wildcards if appropriate (e.g. Name LIKE '%John%').
        
        User Query: {query}
        Chat History: {chat_history}
        
        Return ONLY the SOQL query.
        """
        
        soql_response_obj = await llm.ainvoke(soql_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(soql_response_obj))
        soql_query = soql_response_obj.content.strip().strip("`").strip()
        
        if soql_query.lower().startswith("soql"):
            soql_query = soql_query[4:].strip()
            
        logger.info(f"[SALESFORCE_TOOL] Generated SOQL: {soql_query}")
        
        # 3. Query Salesforce
        records = query_salesforce(soql_query)
        
        # Remove attributes key for formatting
        for r in records:
            r.pop("attributes", None)
            
        records_summary = json.dumps(records, indent=2) if records else "No matching customer records found."
        
        # Print the Salesforce results directly to the terminal for tracking
        print(f"\n[TRACKING] === SALESFORCE TOOL RESULT ===")
        print(f"SOQL Query: {soql_query}")
        print(f"Records Returned count: {len(records)}")
        print(f"Records sample (first 3): {json.dumps(records[:3], indent=2)}")
        print(f"==========================================\n")
        
        # 4. Format findings via LLM
        format_prompt = f"""
        You are a Client Management Specialist. Format the following retrieved Salesforce customer records into a professional, human-readable markdown response.
        
        Requirements:
        - Use tables or clear bullet points.
        - Detail contact info, status, and related identifiers.
        - If no records are found, state that clearly and suggest details to look for.
        - Do NOT output any code or queries.
        
        User Query: {query}
        Salesforce SOQL Executed: {soql_query}
        Retrieved Records:
        {records_summary}
        """
        
        format_response = await llm.ainvoke(format_prompt)
        _accumulate_usage(total_usage, _extract_token_usage(format_response))
        return format_response.content, total_usage
        
    except Exception as e:
        logger.error(f"[SALESFORCE_TOOL] Tool error: {str(e)}")
        return f"I encountered an error while searching Salesforce: {str(e)}. Please check your Salesforce configurations.", total_usage