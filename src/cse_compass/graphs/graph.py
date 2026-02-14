"""CSE Compass LangGraph - stock analysis pipeline."""

from typing import Annotated, Any, List, Set

import asyncio
import json,re
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages import AIMessage

from cse_compass.tools.email_tool import get_email_tool
from cse_compass.tools.mcp_cse import get_cse_tools
from cse_compass.tools.rag_tool import get_rag_search_tool
from cse_compass.tools.web_search import get_web_search_tool

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    company_name: str
    ticker_from_company: str
    intent: str
    buy_price: float
    rag_search_results: str
    web_search_results: dict
    web_search_done: bool


cse_tools = get_cse_tools()
web_search_tool = get_web_search_tool()
rag_search_tool = get_rag_search_tool()
email_tool = get_email_tool()

tools = [*cse_tools, web_search_tool, rag_search_tool, email_tool]

rag_summarizer_llm = ChatOpenAI(model="gpt-4o-mini")

RAG_QUERIES = [
    "Revenue trend",
    "Profitability (net profit / operating profit / margins)",
    "Debt / borrowings / leverage",
    "Cash & liquidity (cash balance, current assets/liabilities)",
    "Cash flow quality (operating cash flow vs profit, capex)",
    "Management outlook / forward-looking commentary",
    "Key risks mentioned (FX, interest rates, sector risks, regulation)",
]


def rag_summarizer_node(state: State) -> State:
    company_name = state.get("company_name")
    ticker = state.get("ticker_from_company")
    if not company_name or not ticker:
        return {"rag_search_results": "Company name or ticker missing. RAG retrieval not executed."}

    retrieved_blocks: List[str] = []

    # 1) Deterministic multiple tool calls
    for topic in RAG_QUERIES:
        q = f"{topic} for {company_name} ({ticker}). Extract only what is explicitly stated in the report."
        tool_out = rag_search_tool.invoke({"query": q, "ticker": ticker})
        # tool_out could be str or dict; normalize:
        if isinstance(tool_out, dict):
            text = tool_out.get("text") or tool_out.get("result") or str(tool_out)
        else:
            text = str(tool_out)

        retrieved_blocks.append(f"### {topic}\n{text}".strip())

    combined = "\n\n".join(retrieved_blocks)

    # 2) One LLM call to summarize evidence (no tool calls here)
    system = f"""
        "You summarize financial report excerpts for a single company. "
        "Use only the provided excerpts; do not invent numbers. "
        "If a topic has no evidence, say 'Not mentioned in retrieved excerpts.' "
        "Output a compact structured summary with bullet points under headings."
    

    Summarize the following financial report excerpts for a single company.
        Company: {company_name}
        Ticker: {ticker}

        EXCERPTS (by topic):
        {combined}

        Write:
        - Revenue trend:
        - Profitability:
        - Debt/borrowings:
        - Cash & liquidity:
        - Cash flow quality:
        - Outlook:
        - Key risks:
        Keep it concise (max ~250-300 words).
        """

    response = rag_summarizer_llm.invoke([SystemMessage(content=system)])
    new_state = {
        "messages": state["messages"],
        "company_name": state["company_name"],
        "ticker_from_company": state["ticker_from_company"],
        "intent": state["intent"],
        "buy_price": state["buy_price"],
        "rag_search_results": response.content,
        "web_search_results": state["web_search_results"],
        "web_search_done": state["web_search_done"],
    }
    return new_state


def rag_search_node_router(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "rag_tools"
    return "cse_mcp_node"


cse_mcp_llm = ChatOpenAI(model="gpt-4o-mini")
cse_mcp_llm_with_tools = cse_mcp_llm.bind_tools(cse_tools)


def cse_mcp_node(state: State) -> dict:
    ticker = state.get("ticker_from_company", "").strip()

    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    tool_count = len(tool_msgs)

    if tool_count == 0:
        sys = f"""
            Call EXACTLY ONE tool: get_stock_data
            Input: symbol = {ticker}
            Return ONLY the tool call. Do not summarize. Do not call any other tool.
            """
        resp = cse_mcp_llm_with_tools.invoke([SystemMessage(content=sys)])
        return {"messages": [resp],}

    if tool_count == 1:
        sys = f"""
            Call EXACTLY ONE tool: get_detailed_company_info
            Input: symbol = {ticker}
            Return ONLY the tool call. Do not summarize. Do not call any other tool.
            """
        resp = cse_mcp_llm_with_tools.invoke([SystemMessage(content=sys)])
        return {"messages": [resp],}

    stock_payload = tool_msgs[-2].content
    detail_payload = tool_msgs[-1].content

    system = """
        You will be given TWO tool outputs (JSON).
        Create a concise bullet list summary.

        Required sections:
        - Price snapshot: price, change, changePercentage, lastUpdated
        - 52-week range: high, low
        - Activity: ytdShareVolume, ytdTurnover
        - Market cap: marketCap
        - Other: sharesIssued, beta (if present)

        Rules:
        - Use ONLY provided JSON fields.
        - If missing, write N/A.
        - Do NOT call any tools.
        - Output ONLY the bullet list (no extra text).
        """
    user = f"""
        TOOL OUTPUT 1 (get_stock_data):
        {stock_payload}

        TOOL OUTPUT 2 (get_detailed_company_info):
        {detail_payload}
        """

    resp = cse_mcp_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    return {  
        "messages": [resp], 
    }


def cse_mcp_node_router(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "cse_mcp_tools"
    return "web_search_node"

def tool_call_ids_for_tool(messages: List[Any], tool_name: str) -> Set[str]:
    ids = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (getattr(m, "tool_calls", None) or []):
                # tc is usually dict-like with "name" and "id"
                if tc.get("name") == tool_name and tc.get("id"):
                    ids.add(tc["id"])
    return ids

def tool_messages_for_tool(messages: List[Any], tool_name: str) -> List[ToolMessage]:
    ids = tool_call_ids_for_tool(messages, tool_name)
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            if getattr(m, "tool_call_id", None) in ids:
                out.append(m)
            # fallback if name exists
            elif getattr(m, "name", None) == tool_name:
                out.append(m)
    return out

def _parse_tool_content(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return content
    return content


web_search_llm = ChatOpenAI(model="gpt-4o-mini")
web_search_llm_with_tools = web_search_llm.bind_tools([web_search_tool])


def web_search_node(state: State) -> dict:
    company = (state.get("company_name") or "").strip()

    if state.get("web_search_done") or state.get("web_search_results"):
        return {}

    web_tool_msgs = tool_messages_for_tool(state["messages"], "web_search")

    if len(web_tool_msgs) < 4:
        sys = f"""
            Call the web search tool exactly 4 times (one query per call) and nothing else:

            1) "{company} Colombo Stock Exchange news"
            2) "{company} earnings OR annual report highlights OR results"
            3) "Sri Lanka interest rates inflation exchange rate latest"
            4) "global risks oil prices shipping disruption war impact markets"

            Rules:
            - Return ONLY tool calls.
            - Do not summarize yet.
            """
        resp = web_search_llm_with_tools.invoke([SystemMessage(content=sys)])
        return {"messages": [resp],}

    payloads = [_parse_tool_content(m.content) for m in web_tool_msgs[-4:]]

    system = """
        You will receive 4 web search result payloads (one per query).
        Return ONLY structured and summarized data in this exact schema:

        {
            "company_news": [{"title":"","snippet":"","url":"","date":""}],
            "company_fundamentals_news": [{"title":"","snippet":"","url":"","date":""}],
            "sri_lanka_macro": [{"title":"","snippet":"","url":"","date":""}],
            "global_risks": [{"title":"","snippet":"","url":"","date":""}]
        }
        

        Rules:
        - Use only provided payloads; do not invent URLs/dates.
        - If missing, put "unknown".
        - Keep max 3 items per category.
        - Do NOT call any tools.
        """
    user = f"""
        PAYLOAD 1 (company news):
        {payloads[0]}

        PAYLOAD 2 (fundamentals/earnings):
        {payloads[1]}

        PAYLOAD 3 (Sri Lanka macro):
        {payloads[2]}

        PAYLOAD 4 (global risks):
        {payloads[3]}
        """

    resp = web_search_llm_with_tools.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    results = _safe_json(resp.content)

    return {
        "web_search_results": results,
        "web_search_done": True,
        "messages": [resp],
    }


def web_search_node_router(state: State) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "web_search_tools"
    return "cse_compass_report_node"


cse_compass_llm = ChatOpenAI(model="gpt-4o-mini")

def _safe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def cse_compass_report_node(state: State) -> State:
    company_name = state.get("company_name")
    system_message = f"""
        You are CSE Compass, an evidence-based stock analysis assistant for the Colombo Stock Exchange.

        Your task:
        Generate a concise and more detailed investment report and recommendation using ONLY the provided inputs.

        Rules:
        - Do NOT invent facts, numbers, dates, or events.
        - If information is missing, explicitly say "Data not available".
        - Be neutral and cautious.
        - No tool calls..

        Output MUST follow this structure exactly:

        Title: {company_name} Analysis Report for stock trading

        Market Snapshot:
        Financial Signals:
        News & Macro Context:
        Risks:
        Watchlist:
        Recommendation + Confidence, based on the anlyses you should recommend to {state.get("intent")} the stock.
        Disclaimer:

        Confidence must be a number between 0.0 and 1.0.
        to the end add a disclaimer: "please note this is a AI generated report to assist you in your investment decisions. It is not financial advice and you should consult with a financial advisor before making any investment decisions.".
        If you have web search reuslurls, add them as well to the report.
        """
    user_message = f"""
        I want to buy or sell {company_name} stock. Give me a recommendation based on the provided information: {state.get("messages")} 
        and {state.get("rag_search_results")} and {state.get("web_search_results")}.
        """

    resp = cse_compass_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ])


    return {
        "messages": [resp],
    }

def send_email_node(state: State) -> dict:
    if state.get("email_sent"):
        return {}

    last = state["messages"][-1] if state.get("messages") else None
    if last is None:
        return {"email_sent": False}

    body = last.content if hasattr(last, "content") else str(last)

    email_tool.invoke({"body": body})

    return {"email_sent": True}


graph_builder = StateGraph(State)
graph_builder.add_node("rag_summarizer_node", rag_summarizer_node)
graph_builder.add_edge(START, "rag_summarizer_node")
graph_builder.add_edge("rag_summarizer_node", "cse_mcp_node" )

graph_builder.add_node("cse_mcp_node", cse_mcp_node)
graph_builder.add_node("cse_mcp_tools", ToolNode(tools=cse_tools))
graph_builder.add_conditional_edges("cse_mcp_node", cse_mcp_node_router, {"cse_mcp_tools": "cse_mcp_tools", "web_search_node": "web_search_node"})
graph_builder.add_edge("cse_mcp_tools", "cse_mcp_node")

graph_builder.add_node("web_search_node", web_search_node)
graph_builder.add_node("web_search_tools", ToolNode(tools=[web_search_tool]))
graph_builder.add_conditional_edges("web_search_node", web_search_node_router, {"web_search_tools": "web_search_tools", "cse_compass_report_node": "cse_compass_report_node"})
graph_builder.add_edge("web_search_tools", "web_search_node")

graph_builder.add_node("cse_compass_report_node", cse_compass_report_node)
graph_builder.add_edge("cse_compass_report_node", "send_email_node")
graph_builder.add_node("send_email_node", send_email_node)
graph_builder.add_edge("send_email_node", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def make_thread_id() -> str:
    return str(uuid.uuid4())

state = {
        "messages": [HumanMessage(content="I want to buy ROYAL CERAMICS LANKA PLC stock. Give me a recommendation.")],
        "company_name": "ROYAL CERAMICS LANKA PLC",
        "ticker_from_company": "RCL.N0000",
        "intent": "BUY",
        "buy_price": None,
        "rag_search_results": "",
        "web_search_results": {},
        "web_search_done": False
}
config = {"configurable": {"thread_id": make_thread_id()}}
result = asyncio.run(graph.ainvoke(state, config=config))
last_msg = result["messages"][-1]
print(last_msg.content if hasattr(last_msg, "content") else last_msg)
