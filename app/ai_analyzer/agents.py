from .tools import get_schema_and_sample, get_columns_list
from typing import Tuple, Dict, List

def _extract_token_usage(response) -> Dict[str, int]:
    """Extract token usage from LLM response."""
    usage = getattr(response, "usage_metadata", {}) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0)
    }

async def summarize_history(llm, history: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    """
    Summarizes older conversation history to preserve context without exceeding token limits.
    """
    if not history:
        return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    history_text = "\n".join([f"User: {item.get('user','')}\nAssistant: {item.get('ai','')}" for item in history])
    
    prompt = f"""
    Summarize the following conversation history between a User and a Professional Financial Data Analyst.
    Preserve key details:
    - The user's initial intent/goal (e.g., "analyzing sales data").
    - Key questions asked (e.g., "first query was about column count").
    - Important findings or decisions made.
    - Any specific preferences stated by the user.
    
    Keep the summary concise (max 3-4 sentences).
    
    Conversation:
    {history_text}
    """
    
    response = await llm.ainvoke(prompt)
    usage = _extract_token_usage(response)
    return response.content.strip(), usage

async def parse_query_to_tool(llm, query: str, conversation: list, db_path: str, context_summary: str = "") -> Tuple[str, Dict[str, int]]:
    """
    Enhanced router with intelligent decision tree logic that correctly routes queries to appropriate tools.
    Now supports context_summary for long-term memory.
    
    Key improvements:
    1. "Do we have...", "Are there..." specific factual questions → query_dataset (not detect_anomalies)
    2. Tracks clarification history to prevent repeated clarifications on same topic
    3. Better keyword matching for each tool
    4. Distinguishes between specific data retrieval vs generic analysis
    5. Handles "all charts", "suggest visualization" -> generate_visualization
    6. Handles "python program" requests -> routes to data tool (not conversational)
    
    Returns:
      - (One tool name from the allowed list, token_usage)
      - OR (a clarification question starting with 'Could you clarify', token_usage)
      - OR ('conversational', token_usage)
    """

    # Load dataset preview/context if available
    sample_context = ""
    columns_info = ""
    if db_path:
        try:
            sample_context = get_schema_and_sample(db_path, sample_size=10)
            columns_list = get_columns_list(db_path)
            columns_info = f"\nAvailable columns: {', '.join(columns_list)}"
        except Exception:
            sample_context = "(Dataset unavailable or could not be loaded.)"
            columns_info = ""

    # Build conversation history text
    chat_history = ""
    if context_summary:
        chat_history += f"[Previous Conversation Summary]: {context_summary}\n\n"
    
    if conversation:
        chat_history += "\n".join([f"User: {c.get('user','')}\nAssistant: {c.get('ai','')}" for c in conversation])

    # Allowed tools (must match exactly these keys)
    allowed_tools = [
        "get_data_summary",
        "detect_anomalies",
        "generate_insights",
        "query_dataset",
        "generate_detailed_report",
        "generate_visualization",
        "conversational"
    ]

    # Long-form, strict instruction for the LLM to:
    # - Decide whether query is conversational, actionable (tool), or underspecified (clarify)
    # - If underspecified, produce a single, specific clarification question (LLM crafts it)
    # - If actionable, choose the best tool name exactly from allowed_tools
    # - Consider dataset schema/sample + conversation context when deciding
    # - Produce output in a strictly machine-readable single-line format (no extra text)
    # - Use decision tree logic to route correctly
    
    
    prompt = f"""
    You are the Financial Intelligence Router for the Finance Platform. Your job is to examine a user's natural-language financial or data query and decide precisely one of the following single-line outputs:

    1) The exact name of one of the data-processing tools:
    get_data_summary
    detect_anomalies
    generate_insights
    query_dataset
    generate_detailed_report
    generate_visualization

    2) The single word: conversational
    (use this for greetings, chit-chat, and user prompts that are unrelated to dataset analysis)

    3) A single clarifying question like could you clarify ... ?
    (This means the user's query is underspecified or ambiguous; the question should be the minimal, focused follow-up needed to proceed.)

    ensure that the clarification question should not be too vague add some options or choices to make it more specific
    DECISION TREE ROUTING LOGIC:
    ============================
    
    STEP 1: IS IT CONVERSATIONAL?
    - Greetings (hello, hi, thanks, no, bye)
    - Off-topic chit-chat (small talk, meta-discussion)
    - No dataset reference
    - **EXCEPTION**: If user asks "generate a python program" or "write code" to analyze data, DO NOT return conversational. Route to the appropriate data tool (e.g., query_dataset or generate_insights).
    → Return: conversational
    
    STEP 2: SPECIFIC FACTUAL QUESTIONS → query_dataset (NOT anomalies)
    - Starts with: "Do we have...", "Are there...", "Is there...", "Can you find..."
    - Asks: "Who/Which/What/Where specific people/records"
    - Commands: "List/Show/Give me/Tell me/Find [specific thing]"
    - **Code Requests**: "Write a python program to calculate average sales" -> Treat as "Calculate average sales" -> query_dataset (or generate_insights if complex).
    - Examples that MUST route to query_dataset:
      * "Do we have anyone managing themselves?" → Check WHERE employee_id = manager_id
      * "Are there any contracts employees in sales?" → Filter query
      * "List employees named John" → Direct data retrieval
      * "Give me details of person X" → Direct lookup
      * "Who is the oldest employee?" → Aggregation query
      * "Python program for average sales" → query_dataset (to get the value)
    → Return: query_dataset
    
    STEP 3: ANOMALY/DATA QUALITY/ISSUES (But verify it's not a specific check above)
    - Asks about: "Any issues/problems/inconsistencies/duplicates/nulls/outliers/anomalies"
    - Wants: General data quality scan, NOT a specific business question
    - Examples:
      * "Detect any problems in the data" → detect_anomalies ✓
      * "Are there duplicate records?" → detect_anomalies ✓ (if asking generally)
      * "Check for missing values" → detect_anomalies ✓
      * BUT: "Do we have anyone managing themselves?" → query_dataset ✓ (specific check, not general anomaly)
    → Return: detect_anomalies (ONLY if it's not a specific business question)
    
    STEP 4: INSIGHTS/COMPARISON/GROUPING/TRENDS → generate_insights
    - Asks: "Compare X and Y", "Which team/dept/group...", "Top/Bottom/Highest/Lowest/Most"
    - Wants: Aggregation, grouping, trends, rankings
    - Examples:
      * "Which manager has most employees?" → Grouping/ranking
      * "Compare contract vs permanent staff" → Comparison
      * "Salary trends over time" → Trends
    → Return: generate_insights
    
    STEP 5: VISUALIZATION/CHART → generate_visualization
    - Keywords: "chart/graph/plot/visualize/dashboard/pie/bar/trends"
    - **Suggestions**: "Suggest a financial chart", "Visualize my revenue", "Show trends"
    - Specifies columns or intent (e.g., "ROI chart", "Budget vs Actual")
    - Examples:
      * "Chart revenue by region" → Visualization
      * "Plot expense distribution" → Visualization
      * "Suggest a good chart for profit margins" → Visualization
    → Return: generate_visualization
    
    STEP 6: FULL REPORT → generate_detailed_report
    - Keywords: "full report", "comprehensive analysis", "detailed report", "everything about"
    - Wants: Complete, multi-section analysis
    → Return: generate_detailed_report
    
    STEP 7: SUMMARY/OVERVIEW → get_data_summary
    - Keywords: "summary", "overview", "describe", "what's in this data", "tell me about"
    - Wants: High-level dataset description
    → Return: get_data_summary
    
    STEP 8: CLARIFICATION HISTORY CHECK
    - IF the previous assistant message was a clarification question
    - AND the current user message provides ANY answer/specificity
    - Then: Proceed with tool, do NOT ask the same clarification again
    - Example: 
      * AI: "Could you clarify which columns to visualize?"
      * User: "maybe sales and revenue"
      * Router should: Proceed to generate_visualization, NOT ask for clarification again
    
    STEP 9: DEFAULT → ASK FOR CLARIFICATION
    - If query is ambiguous after all above checks
    - Ask focused, one-line clarification
    - Example: "Could you clarify whether you want a summary, specific records, or a chart?"

    IMPORTANT OUTPUT RULES:
    - Return EXACTLY ONE line, nothing else (no JSON, no explanation). avoid amiable
    - If returning a tool name, return the tool name alone (exactly as listed).
    - If returning a clarification, start with "Could you clarify"
    - Do NOT include any commentary, examples, or additional lines.
    - If you are uncertain, prefer asking a targeted clarification
    - NEVER ask a clarification if the previous message was already a clarification on the same topic

    CONTEXT YOU HAVE:
    - User Query: {query}
    - Conversation History (recent): 
    {chat_history}

    - Dataset preview / schema sample (if available):
    {sample_context}
    {columns_info}

    CRITICAL EXAMPLES FOR CORRECT ROUTING:
    =====================================
    
    Example 1: "Do we have anyone managing themselves?"
    - NOT anomaly detection (wrong: route to detect_anomalies)
    - IS a specific yes/no check about data (correct: route to query_dataset)
    - Why: User wants to know if employee_id == manager_id exists, not generic anomalies
    - SQL: SELECT * FROM data WHERE employee_id = manager_id
    → Return: query_dataset
    
    Example 2: "Which manager has the most employees?"
    - NOT specific retrieval (wrong: query_dataset)
    - IS grouping and ranking (correct: generate_insights)
    → Return: generate_insights
    
    Example 3: "Detect any problems in the data"
    - IS general data quality scan (correct: detect_anomalies)
    - NOT specific business question
    → Return: detect_anomalies
    
    Example 4: "Compare contract staff vs permanent staff performance"
    - IS comparison and grouping (correct: generate_insights)
    - Groups by employment_type and compares performance
    → Return: generate_insights
    
    Example 5: "Show me employees in sales department"
    - IS specific data retrieval (correct: query_dataset)
    - NOT a grouping/insight question
    → Return: query_dataset
    
    Example 6: "Create a chart showing salary by department"
    - IS visualization request (correct: generate_visualization)
    - Specifies: chart + columns (salary, department)
    → Return: generate_visualization
    
    Example 7: "Tell me about the dataset"
    - IS overview/summary (correct: get_data_summary)
    → Return: get_data_summary
    
    Example 8 (Clarification History):
    - Previous AI: "Could you clarify which specific metrics you want to compare?"
    - User says: "maybe salary and performance"
    - Current query: This IS answering the previous clarification
    - Do NOT ask for clarification again; proceed with the tool
    → Return: generate_insights (or appropriate tool based on new information)
    
    Example 9: "Chart something"
    - UNDERSPECIFIED: Doesn't say what to chart
    - IS ambiguous: Need column names or chart type
    → Return: Could you clarify what you'd like to visualize and which columns to use?
    
    Example 10: "Do we have duplicates?"
    - IS asking about data quality generally (correct: detect_anomalies)
    - NOT a specific business question like "Do we have [specific condition]?"
    → Return: detect_anomalies

    Example 11: "Generate all possible charts"
    - IS visualization request (correct: generate_visualization)
    - User wants suggestions/auto-generation
    → Return: generate_visualization

    Example 12: "Write a python program to calculate average sales"
    - IS data query (correct: query_dataset)
    - User wants the result, phrased as a code request
    → Return: query_dataset

    COLUMN MISMATCH HANDLING:
    - If the user mentions a column that doesn't exist in the dataset, still route to 'query_dataset' tool.
    - The query_dataset tool will intelligently detect the mismatch and ask for clarification with similar column suggestions.
    - Example: User says "List engineers by role" but column is "job_title" → query_dataset will ask "Could you clarify: I don't see 'role', but found similar columns like 'job_title', 'position'. Which one do you mean?"

    Maintain the conversation history and ask for the clarification for example 
    
    user: i have uploaded a dataset what can you with that
    AI : That's great! Once your dataset is uploaded, I can help you explore it by summarizing its key aspects, creating helpful charts, and even generating detailed reports to uncover insights. Just let me know what you'd like to find out!

    user : i have already uploaded
    Ai : Could you clarify What would you like to do with the dataset? For example, would you like a summary, a report, or a visualization?

    User : ur wish
    AI : Could you clarify what you'd like to do with the dataset?

    User: may be a visualisation
    AI : Could you clarify what you would like to visualize, such as which columns or what type of chart?

    User : quantity sold
    Ai : Could you clarify what type of chart you would like for quantity sold, or if you want to group it by another column?

    here the Ai is repeatedly asking for clarification , but the AI get the intent from the previous user response , so dont make the clarfication much repeatedly, you can ask clariification only once or twice based on the context of the conversation history to get clear in the user intent , dont make it repeadted for long

    DECISION GUIDELINES & WHAT COUNTS AS 'UNDERSPECIFIED': for example 
    - Underspecified examples (you should ask a clarification):
    - Missing which columns/metrics (e.g., "visualize sales" — which column is 'sales' or what aggregation?)
    - Missing timeframe or grouping (e.g., "show trend" — which date column and what period?)
    - Missing aggregation method (e.g., "top 10" — by what metric?)
    - Unclear output format (e.g., "generate a report" — summary vs full vs anomalies vs presentation?)
    - Conflicting requests (e.g., "plot profit and show anomalies" — ask which first or whether both in one)
    - Very broad queries that could be multiple tools (e.g., "analyze dataset" → ask what kind of analysis)

    - When generating a clarification question:
    - Be concise, one sentence, polite, and specific.
    - Include examples only if they help make it in 2-3 lines , dont make it as long sentences.
    - Prefer asking about columns, metrics, timeframe, filters, aggregation, or chart type as relevant.
    - If dataset context indicates likely columns, reference the column names to make question precise. Example phrasing:
        Could you clarify whether you want sales summed or averaged and which date column to use?

    - When choosing a tool:
    - If the user explicitly requests 'report' or 'full report' and enough context, return generate_detailed_report.
    - If user asks to 'detect nulls', 'missing', 'inconsistent' or 'outliers', return detect_anomalies.
    - If user asks for 'summary', 'overview', 'describe dataset' or similar, return get_data_summary.
    - If user asks for 'trends', 'top', 'group by', 'compare groups', return generate_insights.
    - If user asks for a chart/visualization or show me the with enough detail (specifies columns or intent), return generate_visualization.
    - If user asks a direct factual query about dataset values (e.g., 'what is the highest X', 'show orders from city Y'), return query_dataset.
    - Use dataset schema and conversation history to disambiguate ambiguous words (e.g., if 'sales' matches a column, prefer that interpretation).

    EDGE CASES:
    - If the query mentions multiple tasks in one sentence, and it's ambiguous whether to do them sequentially or together, ask a clarifying question (start with 'Could you clarify').
    - If the user requests something that would require code execution or revealing backend details, return conversational only if it's purely chat; otherwise ask clarification about the user's intent.
    - If the dataset preview is missing and the query refers to columns (e.g., "plot revenue by city") — ask for clarification or ask the user to upload a dataset; prefer a clarification in one line.
    - Do not fabricate column names or results. Use dataset context only to make clarification precise.

    RETURN FORMAT EXAMPLES (these are examples only — DO NOT output them):
    - If you decide: generate_visualization
    - If you decide: conversational
    - If you decide to clarify: Could you clarify whether you want a bar chart or a line chart and which columns should be used for x/y?

    Now decide and return EXACTLY ONE LINE according to the rules above. just the tool name 
    """

    # Call the LLM
    response = await llm.ainvoke(prompt)
    token_usage = _extract_token_usage(response)
    result = response.content.strip().splitlines()[0].strip() if response.content else ""

    # Sanitise result: only allow exact tool names, 'conversational', or clarification starting with 'Could you clarify'
    if result in allowed_tools:
        return result, token_usage

    if result.lower() == "conversational":
        return "conversational", token_usage

    if result.startswith("Could you clarify"):
        return result, token_usage

    # Secondary safety: Sometimes model returns a clarification but with slight variations (e.g., "Could you clarify..." lowercase or punctuation).
    # Normalize common variants by forcing the prefix if intent appears clarifying.
    lowered = result.lower()
    if lowered.startswith("could you") or lowered.startswith("could you clarify") or lowered.startswith("can you clarify") or lowered.endswith("?"):
        # try to craft a normalized clarification that preserves user intent found in the model output
        # keep as short as possible:
        cleaned = result
        # Ensure it begins with exact prefix:
        if not cleaned.startswith("Could you clarify"):
            # Try to rephrase minimally:
            cleaned = "Could you clarify " + cleaned.lstrip("Could you").lstrip("could you").lstrip(": ").strip()
        # Final safety: single line, no newlines
        cleaned = cleaned.splitlines()[0].strip()
        return cleaned, token_usage

    # Final fallback: safe, minimal clarification
    return "Could you clarify what exactly you'd like me to do with the dataset (e.g., which columns/metrics, aggregation, timeframe, or chart type)?", token_usage