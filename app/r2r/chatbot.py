import re
import os
import io
import base64
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ddgs import DDGS
import urllib.parse
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
# from app.services.llm import call_openrouter_ai as llm_call 
# from app.services.llm import call_groq_ai as llm_call
# from app.services.llm import call_mistral_ai as llm_call
from app.services.llm import call_gemini_ai as llm_call
# from app.services.llm import call_gptoss_ai as llm_call
from app.r2r.schema import ChatResponse
from app.r2r.prompts.common import COMMON_SYSTEM_MESSAGE
from app.r2r.prompts.router import ROUTER_PROMPT
from app.r2r.prompts.sql_gen import SQL_QUERY_PROMPT
from app.r2r.prompts.python_viz import VISUALIZATION_PROMPT, VISUALIZATION_EXPLAIN_PROMPT
from app.r2r.prompts.acknowledgment import ACKNOWLEDGMENT_PROMPT
from app.r2r.prompts.report_gen import REPORT_GEN_PROMPT
from app.r2r.prompts.query_enhancer import ENHANCE_QUERY_PROMPT
from app.r2r.prompts.dashboard import DASHBOARD_PROMPT, DASHBOARD_SQL_GEN_PROMPT
from app.r2r.prompts.market_study import MARKET_STUDY_PROMPT, MARKET_STUDY_SQL_GEN_PROMPT, MARKET_STUDY_SEARCH_GEN_PROMPT, MARKET_STUDY_INTERNAL_SUMMARY_PROMPT, COMPANY_CONTEXT
from app.r2r.prompts.future_prediction import FUTURE_PREDICTION_SQL_PROMPT, FUTURE_PREDICTION_SEARCH_PROMPT, FUTURE_PREDICTION_VIZ_PROMPT, FUTURE_PREDICTION_REPORT_PROMPT
from app.config.database import client

# SQLite Setup
import sqlite3
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "db", "finance_data.db")

# --- DB HELPERS ---
import datetime
from bson import ObjectId

# Mapping MongoDB collections to SQLite tables by Database
COLLECTIONS_MAP = {
    "BP": {
        "products": "products",
        "purchase_orders": "purchase_orders",
        "vendors": "vendors",
        "stores": "stores",
        "inventory": "inventory",
        "vendor_issues": "vendor_issues",
        "recommendations": "recommendations"
    }
}

async def sync_mongo_to_sql():
    """Syncs specified MongoDB collections from multiple databases to local SQLite tables."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        
        sync_results = []
        
        def clean_val(v):
            if isinstance(v, (ObjectId, datetime.datetime)):
                return str(v)
            if isinstance(v, list):
                return [clean_val(i) for i in v]
            if isinstance(v, dict):
                return {k: clean_val(val) for k, val in v.items()}
            return v

        def flatten_json(y):
            out = {}
            def flatten(x, name=''):
                if isinstance(x, dict):
                    # Handle MongoDB-specific internal types
                    if len(x) == 1 and list(x.keys())[0].startswith('$'):
                        val = list(x.values())[0]
                        key = name[:-1] if name.endswith('_') else name
                        out[key if key else 'id'] = clean_val(val)
                        return
                    for a in x:
                        clean_a = a
                        if name:
                            parent_parts = name.split('_')
                            parent_name = parent_parts[-2].lower() if len(parent_parts) > 1 else name[:-1].lower()
                            
                            if clean_a.lower().startswith(parent_name) and len(clean_a) > len(parent_name):
                                test_a = clean_a[len(parent_name):].lstrip('_')
                                if test_a.lower() not in ['id', 'name', 'type']:
                                    clean_a = test_a
                        
                        flatten(x[a], name + clean_a + '_')
                elif isinstance(x, list):
                    key = name[:-1] if name.endswith('_') else name
                    out[key if key else 'data_list'] = str(clean_val(x))
                else:
                    key = name[:-1] if name.endswith('_') else name
                    out[key if key else 'data_val'] = clean_val(x)
            flatten(y)
            return out

        for db_name, collections in COLLECTIONS_MAP.items():
            print(f"Syncing from Database: {db_name}...")
            current_db = client[db_name]
            
            for mongo_coll, sql_table in collections.items():
                print(f"  - {mongo_coll} -> {sql_table}...")
                collection = current_db[mongo_coll]
                cursor = collection.find({})
                results = await cursor.to_list(length=5000)
                
                if not results:
                    sync_results.append(f"0 docs from {mongo_coll}")
                    continue

                # Process this collection
                flattened_data = [flatten_json(r) for r in results]
                df_flat = pd.DataFrame(flattened_data)
                
                # Check for duplicate column names after flattening
                if df_flat.columns.duplicated().any():
                    cols = pd.Series(df_flat.columns)
                    for dupe in cols[cols.duplicated()].unique():
                        mask = cols == dupe
                        cols[mask] = [f"{dupe}_{i}" if i != 0 else dupe for i in range(mask.sum())]
                    df_flat.columns = cols

                def final_clean(val):
                    if isinstance(val, (dict, list, ObjectId, datetime.datetime)):
                        return str(val)
                    return val

                # Use .map instead of .applymap to avoid FutureWarnings
                df_flat = df_flat.map(final_clean)
                
                # Sanitize column names robustly to avoid SQLite syntax errors
                new_cols = []
                for col in df_flat.columns:
                    clean_col = re.sub(r'[^a-zA-Z0-9_]', '', str(col)).lower()
                    if not clean_col: clean_col = f"col_{len(new_cols)}"
                    new_cols.append(clean_col)
                
                # Final check for duplicate sanitized column names before creating SQL table
                final_cols = []
                for col in new_cols:
                    if col in final_cols:
                        i = 1
                        while f"{col}_{i}" in final_cols:
                            i += 1
                        final_cols.append(f"{col}_{i}")
                    else:
                        final_cols.append(col)
                df_flat.columns = final_cols

                if df_flat.empty:
                    sync_results.append(f"Empty data {sql_table}")
                    continue

                df_flat.to_sql(sql_table, conn, if_exists='replace', index=False)
                sync_results.append(f"{len(df_flat)} docs in {sql_table}")
                print(f"    - Success: {len(df_flat)} rows, {len(df_flat.columns)} columns")

        conn.close()
        return True, "Sync Complete: " + ", ".join(sync_results)
    except Exception as e:
        print(f"Sync Error: {e}")
        return False, str(e)

async def get_sql_context():
    """Returns the schemas and sample rows of all available SQLite tables."""
    try:
        if not os.path.exists(DB_PATH):
            success, msg = await sync_mongo_to_sql()
            if not success: return "", ""

        conn = sqlite3.connect(DB_PATH)
        schema_parts = []
        sample_parts = []
        
        # Flatten all tables in the map
        all_tables = []
        for db_name, collections in COLLECTIONS_MAP.items():
            all_tables.extend(collections.values())

        for sql_table in all_tables:
            try:
                df_sample = pd.read_sql_query(f"SELECT * FROM {sql_table} LIMIT 3", conn)
                schema = pd.read_sql_query(f"PRAGMA table_info({sql_table})", conn).to_string()
                
                schema_parts.append(f"TABLE: {sql_table}\n{schema}")
                sample_parts.append(f"SAMPLE {sql_table}:\n{df_sample.to_csv(index=False)}")
            except:
                continue # Table might not exist yet if sync skipped it
                
        conn.close()
        return "\n\n".join(schema_parts), "\n\n".join(sample_parts)
    except Exception as e:
        print(f"SQL Context Error: {e}")
        return "", ""

def clean_sql_query(sql_response: str) -> str:
    """Robustly extract SQL from LLM response, handling markdown and conversational fluff."""
    # 1. Try to find content within ```sql ... ```
    match = re.search(r'```sql\s*(.*?)\s*```', sql_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().strip(';')
    
    # 2. Try to find content within ``` ... ```
    match = re.search(r'```\s*(.*?)\s*```', sql_response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if "SELECT" in content.upper():
            return content.strip(';')
            
    # 3. Fallback: Take everything starting from SELECT until ; or end of string
    # We use \s+ to ensure it's the SELECT keyword and not just part of a word
    match = re.search(r'(SELECT\s+.*?(?:;|$))', sql_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip().strip(';')
        
    # Final cleanup of any common prefixes
    cleaned = re.sub(r'^(sql\s+|query\s*:?\s*)', '', sql_response, flags=re.IGNORECASE)
    return cleaned.strip().strip(';')

# --- DASHBOARD GENERATOR ---

async def generate_dashboard_data(question: str, schema: str, sample: str, history_text: str):
    """Gathers dynamic aggregate financial data by asking the LLM to generate SQL queries, then uses results to design a dashboard."""
    try:
        # STEP 1: Get dynamic SQL queries from LLM
        sql_gen_prompt = DASHBOARD_SQL_GEN_PROMPT.format(schema=schema, query=question)
        raw_sql_response = await llm_call(sql_gen_prompt)
        print(f"Dashboard SQL Gen Response: {raw_sql_response}")
        
        # Clean markdown if present
        cleaned_sql_response = raw_sql_response.strip()
        if cleaned_sql_response.startswith('```'):
            cleaned_sql_response = re.sub(r'^```(?:json)?\s*', '', cleaned_sql_response)
            cleaned_sql_response = re.sub(r'\s*```$', '', cleaned_sql_response)
            
        sql_queries = json.loads(cleaned_sql_response)
        
        # STEP 2: Execute dynamic SQL queries
        conn = sqlite3.connect(DB_PATH)
        data_context_parts = []
        
        for i, sql in enumerate(sql_queries):
            try:
                df = pd.read_sql_query(sql, conn)
                if not df.empty:
                    data_context_parts.append(f"--- QUERY {i+1} RESULTS ---\n{df.to_string(index=False)}")
            except Exception as e:
                print(f"Dashboard Dynamic SQL Error ({i+1}): {e}\nQuery: {sql}")
                continue
        
        conn.close()
        
        if not data_context_parts:
            print("No data returned from dynamic SQL queries.")
            return None
        
        data_context = "\n\n".join(data_context_parts)
        data_context += f"\n\n--- SCHEMA ---\n{schema}"
        
        prompt = DASHBOARD_PROMPT.format(data_context=data_context, query=question)
        raw_response = await llm_call(prompt)
        print(f"Dashboard LLM Raw Response (first 500 chars): {raw_response[:500]}")
        
        # Clean the response — strip markdown code blocks if present
        cleaned = raw_response.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        dashboard_config = json.loads(cleaned)
        return dashboard_config
        
    except json.JSONDecodeError as e:
        print(f"Dashboard JSON Parse Error: {e}")
        print(f"Raw response was: {raw_response[:1000]}")
        return None
    except Exception as e:
        print(f"Dashboard Generation Error: {e}")
        return None

# --- MARKET STUDY GENERATOR ---

async def generate_market_study_data(question: str, schema: str, history_text: str):
    """Combines internal SQL data with external web search to provide market context."""
    try:
        print("--- STARTING MARKET STUDY ---")
        
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        print(f"Current System Date: {current_date}")
        
        # STEP 1: Get internal data via dynamic SQL
        sql_gen_prompt = MARKET_STUDY_SQL_GEN_PROMPT.format(schema=schema, query=question, company_context=COMPANY_CONTEXT, current_date=current_date)
        raw_sql_response = await llm_call(sql_gen_prompt)
        print(f"Market Study SQL Gen Response: {raw_sql_response}")
        
        cleaned_sql = raw_sql_response.strip()
        if cleaned_sql.startswith('```'):
            cleaned_sql = re.sub(r'^```(?:json)?\s*', '', cleaned_sql)
            cleaned_sql = re.sub(r'\s*```$', '', cleaned_sql)
            
        try:
            sql_queries = json.loads(cleaned_sql)
        except:
            sql_queries = []
            
        internal_data_parts = []
        if sql_queries and isinstance(sql_queries, list):
            conn = sqlite3.connect(DB_PATH)
            for i, sql in enumerate(sql_queries):
                try:
                    df = pd.read_sql_query(sql, conn)
                    if not df.empty:
                        internal_data_parts.append(f"--- INTERNAL DATA SET {i+1} ---\n{df.to_string(index=False)}")
                except Exception as e:
                    print(f"Market Study SQL Error ({i+1}): {e}\nQuery: {sql}")
            conn.close()
            
        internal_data = "\n\n".join(internal_data_parts) if internal_data_parts else "No internal data found or relevant."
        print(f"--- INTERNAL DATA GATHERED ---\n{internal_data}\n------------------------------")

        # STEP 1.5: Summarize Internal Data
        print("Summarizing Internal Data...")
        summary_prompt = MARKET_STUDY_INTERNAL_SUMMARY_PROMPT.format(query=question, internal_data=internal_data, current_date=current_date)
        internal_summary = await llm_call(summary_prompt)
        print(f"--- INTERNAL SUMMARY ---\n{internal_summary}\n------------------------")

        # STEP 2: Get external data via Web Search
        search_gen_prompt = MARKET_STUDY_SEARCH_GEN_PROMPT.format(query=question, company_context=COMPANY_CONTEXT, internal_summary=internal_summary, current_date=current_date)
        search_queries_response = await llm_call(search_gen_prompt)
        
        # Parse JSON queries
        cleaned_search = search_queries_response.strip()
        if cleaned_search.startswith('```'):
            cleaned_search = re.sub(r'^```(?:json)?\s*', '', cleaned_search)
            cleaned_search = re.sub(r'\s*```$', '', cleaned_search)
            
        try:
            search_queries = json.loads(cleaned_search)
        except Exception as e:
            print(f"Failed to parse search queries JSON. Defaulting to single query. Error: {e}")
            search_queries = [search_queries_response[:100]] # Fallback
            
        if not isinstance(search_queries, list):
            search_queries = [str(search_queries)]
            
        print(f"Market Study Web Search Queries: {search_queries}")
        
        external_data_parts = []
        
        for search_query in search_queries:
            search_query = str(search_query).strip()
            print(f"Executing Search for: {search_query}")
            
            # 2a. DuckDuckGo Search (DDGS)
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=3)) # 3 per query
                    for res in results:
                        external_data_parts.append(f"[DDG] Query: {search_query} | Source: {res.get('title', 'Unknown')} ({res.get('href', '')})\nSnippet: {res.get('body', '')}")
            except Exception as e:
                print(f"Market Study DDG Error for '{search_query}': {e}")
                
            # 2b. SerpAPI Search
            serp_key = os.getenv("SERPAPI_API_KEY")
            if serp_key and GoogleSearch:
                try:
                    params = {
                        "engine": "google",
                        "q": search_query,
                        "api_key": serp_key,
                        "num": 3
                    }
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    if "organic_results" in results:
                        for res in results["organic_results"]:
                            external_data_parts.append(f"[SerpAPI] Query: {search_query} | Source: {res.get('title', 'Unknown')} ({res.get('link', '')})\nSnippet: {res.get('snippet', '')}")
                except Exception as e:
                    print(f"Market Study SerpAPI Error for '{search_query}': {e}")
                    
        external_data = "\n\n".join(external_data_parts) if external_data_parts else "No external web data available."
        print(f"External Data Gathered ({len(external_data_parts)} snippets found).")

        # STEP 3: Synthesize final report
        final_prompt = MARKET_STUDY_PROMPT.format(
            query=question,
            internal_summary=internal_summary,
            external_data=external_data,
            company_context=COMPANY_CONTEXT,
            current_date=current_date
        )
        
        final_report = await llm_call(final_prompt)
        print("Market Study Final Report Generated.")
        return final_report
        
    except Exception as e:
        print(f"Market Study Generation Error: {e}")
        return None

async def generate_future_prediction_data(question: str, schema: str, history_text: str):
    """Combines internal data with external forecasts to predict future trends with charts."""
    try:
        print("--- STARTING FUTURE PREDICTION ---")
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        
        # 1. Internal Context (Baseline)
        sql_prompt = FUTURE_PREDICTION_SQL_PROMPT.format(schema=schema, query=question)
        sql_resp = await llm_call(sql_prompt)
        sql_queries = []
        try:
            cleaned_sql = sql_resp.strip().strip('`').replace('json', '').strip()
            sql_queries = json.loads(cleaned_sql)
        except: pass
        
        internal_data_parts = []
        if sql_queries:
            conn = sqlite3.connect(DB_PATH)
            for i, sql in enumerate(sql_queries[:2]):
                try:
                    df = pd.read_sql_query(sql, conn)
                    if not df.empty:
                        internal_data_parts.append(df.to_string(index=False))
                except: pass
            conn.close()
        internal_summary = "\n".join(internal_data_parts) if internal_data_parts else "No specific internal historical data found."

        # 2. External Forecasts (Search)
        search_prompt = FUTURE_PREDICTION_SEARCH_PROMPT.format(query=question, current_date=current_date)
        search_resp = await llm_call(search_prompt)
        search_queries = []
        try:
            cleaned_search = search_resp.strip().strip('`').replace('json', '').strip()
            search_queries = json.loads(cleaned_search)
        except: pass
        
        external_data_parts = []
        for q in search_queries:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=3))
                    for res in results:
                        external_data_parts.append(f"Source: {res.get('title')} | URL: {res.get('href')}\nSnippet: {res.get('body')}")
            except: pass
        external_data = "\n\n".join(external_data_parts) if external_data_parts else "No external forecast data available."

        # 3. Generate Predictive Chart
        viz_prompt = FUTURE_PREDICTION_VIZ_PROMPT.format(
            internal_summary=internal_summary,
            external_data=external_data
        )
        viz_code = await llm_call(viz_prompt)
        viz_code = viz_code.strip().strip('`').replace('python', '').strip()
        
        chart_base64 = ""
        try:
            local_env = {"plt": plt, "pd": pd, "np": np}
            exec(viz_code, local_env)
            fig = local_env.get("fig")
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
        except Exception as e:
            print(f"Prediction Chart Error: {e}")

        # 4. Synthesize Final Report
        report_prompt = FUTURE_PREDICTION_REPORT_PROMPT.format(
            query=question,
            current_date=current_date,
            internal_summary=internal_summary,
            external_data=external_data
        )
        final_report = await llm_call(report_prompt)
        
        if chart_base64:
            chart_markdown = f"![Forecast Chart](data:image/png;base64,{chart_base64})"
            # Handle cases where LLM might wrap placeholder in backticks
            if "[CHART_PLACEHOLDER]" in final_report:
                final_report = final_report.replace("[CHART_PLACEHOLDER]", chart_markdown)
            elif "`[CHART_PLACEHOLDER]`" in final_report:
                final_report = final_report.replace("`[CHART_PLACEHOLDER]`", chart_markdown)
            else:
                # Fallback: regex to find any variation including backticks
                final_report = re.sub(r'`?\[CHART_PLACEHOLDER\]`?', chart_markdown, final_report)
            
        print("Future Prediction Report Generated.")
        return final_report

    except Exception as e:
        print(f"Future Prediction Error: {e}")
        return None

# --- CHATBOT LOGIC ---

async def get_chatbot_response(question: str, chat_context: list = None):
    try:
        history_text = "\n".join(chat_context) if chat_context else "[]"
        
        # 0. GET SQL CONTEXT (Schema + 5 Sample Rows)
        sql_ctx = await get_sql_context()
        if not sql_ctx or not sql_ctx[0]:
            return ChatResponse(answer="I couldn't access the data schema. Please refresh the page to re-initialize the analysis engine.", voice_enabled=False)
        
        schema, sample = sql_ctx

        # 0b. ENHANCE QUERY
        enhance_prompt = ENHANCE_QUERY_PROMPT.format(chat_history=history_text, query=question)
        enhanced_question = await llm_call(enhance_prompt)
        enhanced_question = enhanced_question.strip()
        print(f"Enhanced Query: {enhanced_question}")

        # 1. ROUTING
        router_prompt = ROUTER_PROMPT.format(chat_history=history_text, query=enhanced_question)
        route_result = await llm_call(router_prompt)
        route_result = route_result.strip().lower()
        print(f"Routing Decision: {route_result}")
        
        # 2. HANDLE CLARIFICATION
        if "could you clarify" in route_result:
            return ChatResponse(answer=route_result, voice_enabled=False)
            
        # 3. CONVERSATIONAL
        if "conversational" in route_result:
            full_prompt = f"{COMMON_SYSTEM_MESSAGE}\n\nChat History:\n{history_text}\n\nUser: {enhanced_question}"
            answer = await llm_call(full_prompt)
            return ChatResponse(answer=answer, voice_enabled=False)
        
        # 3b. DASHBOARD
        if "generate_dashboard" in route_result:
            dashboard_config = await generate_dashboard_data(question, schema, sample, history_text)
            if dashboard_config:
                # Generate a dynamic summary to provide context for future questions
                kpi_list = dashboard_config.get('kpis', [])
                kpi_summary = ", ".join([f"{k.get('label')}: {k.get('value')}" for k in kpi_list[:3]])
                summary_prompt = f"Summarize these key financial metrics in 2-3 professional sentences to provide immediate context: {kpi_summary}. Mention that the interactive dashboard is ready."
                dynamic_answer = await llm_call(summary_prompt)
                
                return ChatResponse(
                    answer=dynamic_answer,
                    dashboardData=dashboard_config,
                    voice_enabled=False
                )
            else:
                return ChatResponse(
                    answer="I had trouble generating the dashboard metrics. Let me try querying the raw tables instead.",
                    voice_enabled=False
                )
                
        # 3c. MARKET STUDY
        if "market_study" in route_result:
            market_report = await generate_market_study_data(question, schema, history_text)
            if market_report:
                # Use a summary of the report to keep context in chat history
                summary_prompt = f"Provide a very brief (2-sentence) summary of this market study report: {market_report[:1000]}"
                dynamic_ack = await llm_call(summary_prompt)
                return ChatResponse(
                    answer=f"{dynamic_ack}\n\nThe complete Market Study with external research is available in the report panel.",
                    reportData=market_report, # Render it in the report tab seamlessly
                    voice_enabled=False
                )
            else:
                return ChatResponse(
                    answer="I encountered an error while conducting the market study research.",
                    voice_enabled=False
                )

        # 3e. FUTURE PREDICTION
        if "future_prediction" in route_result:
            prediction_report = await generate_future_prediction_data(question, schema, history_text)
            if prediction_report:
                summary_prompt = f"Provide a very brief (2-sentence) forward-looking summary of this future prediction report: {prediction_report[:1000]}"
                dynamic_ack = await llm_call(summary_prompt)
                return ChatResponse(
                    answer=f"{dynamic_ack}\n\nThe complete Future Outlook Report with predictive charts is available in the report panel.",
                    reportData=prediction_report,
                    voice_enabled=False
                )
            else:
                return ChatResponse(
                    answer="I encountered an error while calculating future market predictions.",
                    voice_enabled=False
                )

        # 3d. INSIGHTS GENERATION (Attempt to answer from history first)
        if "generate_insights" in route_result:
            # If the user is asking about insights from a previous dashboard/report, we might already have the data in text context
            insight_check_prompt = f"Using ONLY the provided chat history, can you provide the requested insights? If you have enough data (like KPI values or previous summaries), provide the answer. If you MUST query the database to get new numbers, respond ONLY with 'NEED_SQL'.\n\nHistory:\n{history_text}\n\nQuestion: {enhanced_question}"
            insight_response = await llm_call(insight_check_prompt)
            
            if "NEED_SQL" not in insight_response:
                return ChatResponse(answer=insight_response, voice_enabled=False)
            # Otherwise, fall through to SQL step
            
        # 4. DATA TASKS (SQL Generation & Execution)
        sql_query_input = enhanced_question
        if "generate_insights" in route_result:
             sql_query_input = f"Analyze the data and retrieve key metrics to answer: {enhanced_question}"

        query_prompt = SQL_QUERY_PROMPT.format(schema=schema, sample=sample, query=sql_query_input, chat_history=history_text)
        sql_resp = await llm_call(query_prompt)
        print(f"RAW SQL Response: {sql_resp}")
        
        # Check for model refusal (common when model thinks it needs to 'see' something)
        if "I am an AI" in sql_resp or "cannot access" in sql_resp or "do not have" in sql_resp:
             print("SQL Generator refused. Re-attempting with strict formatting...")
             sql_resp = await llm_call(query_prompt + "\n\nIMPORTANT: You MUST ONLY output the SQL query. DO NOT apologize. DO NOT refuse. If you need data to provide insights, SELECT it from the relevant tables.")
             print(f"Retry SQL Response: {sql_resp}")

        sql_query = clean_sql_query(sql_resp)
        print(f"Executing SQL: {sql_query}")
        
        # Final safety check if cleaning failed to find SQL
        if "SELECT" not in sql_query.upper():
             # Fallback: if it's still text, just return it as a conversational answer
             return ChatResponse(answer=sql_resp, voice_enabled=False)
        
        df = pd.DataFrame()
        table_dict = None
        
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # USER REQUEST: Print the data which comes from the sql for the query
            print("--- SQL QUERY RESULTS ---")
            print(df)
            print("-------------------------")
            
            if not df.empty:
                # Sanitize for frontend
                for col in df.columns:
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                
                table_dict = {
                    "isTable": True,
                    "headers": df.columns.tolist(),
                    "rows": [[str(v) if v is not None else "" for v in row] for row in df.values.tolist()]
                }
            else:
                return ChatResponse(answer="I found no data matching that request on the synced database.", voice_enabled=False)
        except Exception as sql_err:
            print(f"SQL Error: {sql_err}")
            return ChatResponse(answer="I had trouble processing the data. Please try again the question.", voice_enabled=False)
            
        # 4c. Visualization
        chart_data = None
        if "visualization" in route_result or any(word in question.lower() for word in ["chart", "plot", "graph"]):
            viz_prompt = VISUALIZATION_PROMPT + f"\n\nData Context (First 10):\n{df.head(10).to_string()}\n\nUser Query: {question}"
            viz_code = await llm_call(viz_prompt)
            viz_code = re.sub(r'```python\n(.*?)\n```', r'\1', viz_code, flags=re.DOTALL).strip()
            print(f"Executing Viz Code:\n{viz_code}")
            
            try:
                # Prepare data for exec
                # Convert to numeric where possible for better plotting
                df_numeric = df.copy()
                for col in df_numeric.columns:
                    try:
                        df_numeric[col] = pd.to_numeric(df_numeric[col])
                    except:
                        pass
                
                data_list = df_numeric.to_dict(orient='records')
                local_env = {
                    "data": data_list,
                    "df": df_numeric, 
                    "plt": plt, 
                    "matplotlib": matplotlib, 
                    "pd": pd,
                    "np": np
                }
                exec(viz_code, local_env)
                fig = local_env.get("fig")
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    chart_data = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
            except Exception as viz_err:
                print(f"Viz Error: {viz_err}")

        # 4d. Final Answer & Acknowledgment
        final_report = None
        if "generate_detailed_report" in route_result:
            rep_prompt = REPORT_GEN_PROMPT.format(
                query=question,
                data_points=len(df)
            ) + f"\n\nJSON Data Context:\n{df.to_json(orient='records', indent=2)}"
            final_report = await llm_call(rep_prompt)
            summary_answer = "I've generated the detailed report for you. You can find it in the data panel on the right."
        else:
            stats = df.describe().to_string() if not df.empty else "No numeric stats available"
            explain_prompt = VISUALIZATION_EXPLAIN_PROMPT.format(query=question) + f"\n\nResults Summary:\n{df.head(5).to_string()}\nStats:\n{stats}"
            summary_answer = await llm_call(explain_prompt)
            
        ack_prompt = ACKNOWLEDGMENT_PROMPT.format(tool_name=route_result)
        ack_msg = await llm_call(ack_prompt)
        
        final_answer = f"{summary_answer}\n\n{ack_msg}"
        
        return ChatResponse(
            answer=final_answer, 
            tableData=table_dict, 
            chartData=chart_data, 
            reportData=final_report,
            voice_enabled=False
        )
        
    except Exception as e:
        print(f"Global ChatBot Error: {e}")
        return ChatResponse(answer="I'm sorry, I encountered a system error. Please try again later.", voice_enabled=False)

# --- CHUNKED SUMMARIZATION ---
async def summarize_messages(messages: list) -> str:
    """Uses the LLM to summarize a specific chunk of chat history for the frontend."""
    text = "\n".join(messages)
    prompt = f"Summarize the following conversation concisely in 1-2 paragraphs. Capture the main user intents, key facts, and data context so the AI can remember this later:\n\n{text}\n\nSummary:"
    return await llm_call(prompt)