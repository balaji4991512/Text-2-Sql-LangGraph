# %%
import os, asyncio, json, re, sqlite3, pandas as pd
from typing import Annotated, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI

# Use the model you have access to
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
DATA_DIR = "./data"   # your folder with CSVs
SQLITE_DB_PATH = "demo_text_to_sql.db"
print("Using data dir:", DATA_DIR)

# %%
def build_schema_from_csv_folder(data_dir: str):
    catalog = {}
    files = sorted(os.listdir(data_dir))
    for fname in files:
        low = fname.lower()
        if low.endswith(".csv"):
            table = os.path.splitext(fname)[0]
            path = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(path, nrows=5)  # sample to infer columns quickly
                cols = list(df.columns)
            except Exception:
                # fallback: try reading first line
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.readline().strip().split(",")
                    cols = [c.strip() for c in header if c.strip()]
            catalog[table] = {"columns": cols, "sample_rows": None, "source": path}
    return catalog

# %%
SCHEMA_CATALOG = build_schema_from_csv_folder(DATA_DIR)
print("Tables discovered:", list(SCHEMA_CATALOG.keys()))
# quick view of a table
for t,meta in list(SCHEMA_CATALOG.items())[:5]:
    print(t, "->", meta["columns"][:10])

# %%
def init_sqlite_from_csvs(catalog, db_path=SQLITE_DB_PATH):
    conn = sqlite3.connect(db_path)
    for table, meta in catalog.items():
        path = meta["source"]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # sanitize column names
        df.columns = [re.sub(r"\W+", "_", c).lower() for c in df.columns]
        # write to sqlite (replace to refresh)
        df.to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("SQLite DB initialized at", db_path)

init_sqlite_from_csvs(SCHEMA_CATALOG)

# %%
def simple_schema_retrieval(query: str, catalog: dict, n=8):
    # token-match scoring across table names & column names
    tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for t,meta in catalog.items():
        score = 0
        if any(tok in t for tok in tokens):
            score += 3
        for col in meta["columns"]:
            if any(tok in col.lower() for tok in tokens):
                score += 1
        scored.append((score, t))
    scored.sort(reverse=True)
    selected = [t for s,t in scored if s>0][:n]
    if not selected:
        selected = list(catalog.keys())[:min(n, len(catalog))]
    # return a JSON-serializable block
    found = {t: {"columns": catalog[t]["columns"], "source": catalog[t]["source"]} for t in selected}
    return found

def summarize_schema_block_safe(schema_block: dict) -> str:
    # always return plain text summary (no JSON) but we store original JSON separately
    parts = []
    for t, meta in schema_block.items():
        cols = ", ".join(meta["columns"][:20])
        parts.append(f"{t}: columns=[{cols}]")
    return "\n".join(parts)

# %%
async def llm_complete(prompt: str) -> str:
    # try a modern async call; fallback to thread if provider doesn't have ainvoke
    try:
        messages = [{"role":"system","content":prompt}]
        res = await LLM.ainvoke(messages)
        if hasattr(res, "content"):
            return res.content
        if isinstance(res, dict) and "content" in res:
            return res["content"]
        return str(res)
    except AttributeError:
        loop = asyncio.get_running_loop()
        def sync_call():
            messages = [{"role":"system","content":prompt}]
            out = LLM.invoke(messages)
            if hasattr(out, "content"):
                return out.content
            if isinstance(out, dict) and "content" in out:
                return out["content"]
            return str(out)
        return await loop.run_in_executor(None, sync_call)

# %%
# ============================================
# CELL 6 - LangGraph State + 8 Agents (FINAL)
# ============================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    schema_json: str
    schema_text: str
    rewritten_query: str
    join_plan: str
    sql_query: str
    sql_result: Any
    last_sql: str | None
    last_result: Any | None
    conversation_history: list
    error: str | None
    retries: int


# -------------------------------------------------------
# AGENT 1: Schema Retrieval
# -------------------------------------------------------
async def schema_retrieval(state: State):
    # Ensure memory fields exist
    history = state.get("conversation_history") or []
    state["conversation_history"] = history

    q = state["user_query"]
    found = simple_schema_retrieval(q, SCHEMA_CATALOG, n=8)

    state["schema_json"] = json.dumps(found)
    state["schema_text"] = summarize_schema_block_safe(found)

    history.append(
        {"step": "schema", "tables": list(found.keys())}
    )

    return {
        "messages": [{"role": "system", "content": f"Schema retrieval OK: {list(found.keys())}"}],
        "schema_json": state["schema_json"],
        "schema_text": state["schema_text"],
        "conversation_history": history,
        "error": None,
    }


# -------------------------------------------------------
# AGENT 2: Schema Summarizer
# -------------------------------------------------------
async def schema_summarizer(state: State):
    try:
        raw = json.loads(state.get("schema_json", "{}"))
    except Exception:
        raw = {}

    state["schema_text"] = summarize_schema_block_safe(raw)

    return {
        "messages": [{"role": "system", "content": "Schema summary created"}],
        "schema_text": state["schema_text"],
        "error": None,
    }


# -------------------------------------------------------
# AGENT 3: Query Rewriter
# -------------------------------------------------------
async def query_rewriter(state: State):
    history = state.get("conversation_history") or []
    state["conversation_history"] = history
    last_sql = state.get("last_sql") or "None"
    last_result = state.get("last_result") or "None"

    history_text = json.dumps(history, indent=2)

    prompt = f"""
Rewrite the user's query into a precise, schema-grounded intent.

User query:
{state['user_query']}

Conversation history:
{history_text}

Previous SQL:
{last_sql}

Previous Result:
{last_result}

Schema:
{state['schema_text']}

Rewrite as ONE LINE intent that does not depend on vague references.
"""

    out = await llm_complete(prompt)
    rewritten = out.strip().split("\n")[0]

    state["rewritten_query"] = rewritten
    history.append(
        {"step": "rewriter", "rewritten": rewritten}
    )

    return {
        "messages": [{"role": "system", "content": f"Rewritten: {rewritten}"}],
        "rewritten_query": rewritten,
        "conversation_history": history,
        "error": None,
    }


# -------------------------------------------------------
# AGENT 4: Join Planner
# -------------------------------------------------------
async def join_planner(state: State):
    prompt = f"""
You are a join planner.
Rewritten intent: {state['rewritten_query']}
Schema:
{state['schema_text']}

Return:
tables: ...
joins:
tableA.col = tableB.col
"""

    out = await llm_complete(prompt)
    state["join_plan"] = out.strip()

    return {
        "messages": [{"role": "system", "content": "Join plan ready"}],
        "join_plan": state["join_plan"],
        "error": None,
    }


# -------------------------------------------------------
# AGENT 5: SQL Generator
# -------------------------------------------------------
async def sql_generator(state: State):
    history = state.get("conversation_history") or []
    state["conversation_history"] = history

    prompt = f"""
Generate a VALID SQLite SQL query for this intent:
{state['rewritten_query']}

Join plan:
{state['join_plan']}

Schema:
{state['schema_text']}

Return ONLY the SQL. No markdown.
"""

    raw = await llm_complete(prompt)

    cleaned = (
        raw.replace("```sql", "")
           .replace("```SQL", "")
           .replace("```", "")
           .strip()
    )

    match = re.search(r"(?i)(select.*?;)", cleaned, re.S)
    sql = match.group(1).strip() if match else cleaned.strip()

    if not sql.endswith(";"):
        sql += ";"

    state["sql_query"] = sql
    history.append(
        {"step": "sql_generated", "sql": sql}
    )

    return {
        "messages": [{"role": "system", "content": f"SQL generated: {sql}"}],
        "sql_query": sql,
        "conversation_history": history,
        "error": None,
    }


# -------------------------------------------------------
# AGENT 6: Static Analyzer
# -------------------------------------------------------
async def static_analyzer(state: State):
    sql = state.get("sql_query", "")
    errors = []

    # Block dangerous operations
    if re.search(r"\b(drop|delete|update|insert|alter)\b", sql, re.I):
        errors.append("Unsafe operation detected")

    # Simple heuristic — but allow SUM without GROUP BY for now
    # (you can improve this later if needed)

    if errors:
        issue = "; ".join(errors)
        state["error"] = issue
        return {
            "messages": [{"role": "system", "content": issue}],
            "error": issue,
        }

    state["error"] = None
    return {
        "messages": [{"role": "system", "content": "Static analysis OK"}],
        "error": None,
    }


# -------------------------------------------------------
# AGENT 7: SQL Executor (SQLite)
# -------------------------------------------------------
async def executor(state: State):
    history = state.get("conversation_history") or []
    state["conversation_history"] = history

    sql = state.get("sql_query", "")

    if not sql.lower().startswith("select"):
        issue = "Only SELECT queries allowed"
        state["error"] = issue
        return {
            "messages": [{"role": "system", "content": issue}],
            "sql_result": None,
            "error": issue,
        }

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        result = {"columns": cols, "rows": rows}
        conn.close()

        state["sql_result"] = result
        state["last_sql"] = sql
        state["last_result"] = result
        state["error"] = None

        history.append(
            {"step": "execution", "sql": sql, "result": result}
        )

        return {
            "messages": [{"role": "system", "content": "SQL executed"}],
            "sql_result": result,
            "last_sql": sql,
            "last_result": result,
            "conversation_history": history,
            "error": None,
        }

    except Exception as e:
        issue = f"Execution error: {str(e)}"
        state["error"] = issue
        state["sql_result"] = None
        return {
            "messages": [{"role": "system", "content": issue}],
            "sql_result": None,
            "error": issue,
        }


# -------------------------------------------------------
# AGENT 8: Semantic Validator
# -------------------------------------------------------
async def semantic_validator(state: State):
    prompt = f"""
Validate whether SQL result matches user intent.

User query: {state['user_query']}
Rewritten intent: {state.get('rewritten_query')}
SQL:
{state.get('sql_query')}
Result:
{json.dumps(state.get('sql_result'), default=str)}
Schema:
{state.get('schema_text')}

Return STRICT JSON ONLY:
{{
  "is_correct": true/false,
  "issue": "string"
}}
"""

    verdict = await llm_complete(prompt)

    # Robust JSON extraction
    try:
        s = verdict.strip()
        start = s.index("{")
        end = s.rindex("}") + 1
        j = json.loads(s[start:end])
    except Exception:
        # fallback
        if "true" in verdict.lower():
            j = {"is_correct": True, "issue": ""}
        else:
            j = {"is_correct": False, "issue": "Validator could not parse output."}

    if j.get("is_correct", False):
        state["error"] = None
        return {
            "messages": [{"role": "system", "content": "Semantic validation passed"}],
            "error": None,
        }

    else:
        issue = j.get("issue", "Semantic mismatch")
        state["error"] = issue
        return {
            "messages": [{"role": "system", "content": issue}],
            "error": issue,
        }

# %%
def build_graph():
    builder = StateGraph(State)

    # nodes
    builder.add_node(schema_retrieval)
    builder.add_node(schema_summarizer)
    builder.add_node(query_rewriter)
    builder.add_node(join_planner)
    builder.add_node(sql_generator)
    builder.add_node(static_analyzer)
    builder.add_node(executor)
    builder.add_node(semantic_validator)

    # linear flow
    builder.add_edge(START, "schema_retrieval")
    builder.add_edge("schema_retrieval", "schema_summarizer")
    builder.add_edge("schema_summarizer", "query_rewriter")
    builder.add_edge("query_rewriter", "join_planner")
    builder.add_edge("join_planner", "sql_generator")
    builder.add_edge("sql_generator", "static_analyzer")

    # condition after static analyzer
    def static_to_next(state: State):
        return "retry" if state.get("error") else "success"

    builder.add_conditional_edges(
        "static_analyzer",
        static_to_next,
        {
            "success": "executor",
            "retry": "sql_generator",
        }
    )

    # condition after executor
    def exec_to_next(state: State):
        return "retry" if state.get("error") else "success"

    builder.add_conditional_edges(
        "executor",
        exec_to_next,
        {
            "success": "semantic_validator",
            "retry": "sql_generator",
        }
    )

    # condition after validator
    def sem_to_next(state: State):
        return "retry" if state.get("error") else "end"

    builder.add_conditional_edges(
        "semantic_validator",
        sem_to_next,
        {
            "retry": "sql_generator",
            "end": END,
        }
    )

    graph = builder.compile(checkpointer=MemorySaver())
    return graph

# %%
graph = build_graph()
print("Graph compiled correctly!")

# %%
from IPython.display import Image, display
png = graph.get_graph().draw_mermaid_png()
display(Image(png))


# %%
DEFAULT_THREAD = "demo-chat-1"

async def chat(query: str, thread_id: str = DEFAULT_THREAD):
    # ONLY pass new message + user_query
    incoming = {
        "messages": [{"role": "user", "content": query}],
        "user_query": query,
    }

    # thread_id tells LangGraph to load previous memory
    config = {"configurable": {"thread_id": thread_id}}

    # Let LangGraph merge incoming with checkpointed state
    state = await graph.ainvoke(incoming, config=config)

    return {
        "sql": state.get("sql_query"),
        "result": state.get("sql_result"),
        "error": state.get("error"),
        "history": state.get("conversation_history"),
        "last_sql": state.get("last_sql"),
        "last_result": state.get("last_result"),
    }


# # %%
# # Example interactive calls
# resp = await chat("What is total order_amount for 2024?")
# print(resp)

# resp2 = await chat("Break it down by month")
# print(resp2)

# %%



