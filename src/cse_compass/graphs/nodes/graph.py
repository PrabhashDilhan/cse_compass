from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from cse_compass.tools.mcp_cse import get_cse_tools
import gradio as gr
import uuid
import asyncio
from datetime import datetime

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    company_name: str
    ticket_from_company: str
    intent: str
    buy_price: float


cse_tools = get_cse_tools()

ticketfinder_llm = ChatOpenAI(model="gpt-4o-mini")
ticketfinder_llm_with_tools = ticketfinder_llm.bind_tools(cse_tools)

def ticker_finder_node(state: State) -> State:
    messages = state["messages"]
    company_name = state["company_name"]
    ticket_from_company = state["ticket_from_company"]

    response = ticketfinder_llm_with_tools.invoke(messages)
    return {"messages": [response]}

def ticker_finder_router(state: State) -> str:
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "evaluator"

graph_builder = StateGraph(State)
graph_builder.add_node("ticketfinder", ticker_finder_node)
graph_builder.add_edge(START, "ticketfinder")
graph_builder.add_edge("ticketfinder", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def toggle_buy_price(intent: str):
    return gr.update(visible=(intent == "SELL"))

def make_thread_id() -> str:
    return str(uuid.uuid4())

async def process_message(message, company_name: str, intent: str, buy_price: float, thread):
    # Basic UI validation
    if intent not in ("BUY", "SELL"):
        return "❌ Please select BUY or SELL."

    if intent == "SELL":
        if buy_price is None or buy_price <= 0:
            return "❌ For SELL, please enter a valid bought price (LKR per share)."

    config = {"configurable": {"thread_id": thread}}

    state = {
        "messages": message,
        "company_name": company_name,
        "ticket_from_company": None,
        "intent": intent,
        "buy_price": None if intent == "BUY" else buy_price
    }

    result = await graph.ainvoke(state, config=config)

with gr.Blocks(title="CSE Compass") as demo:
    gr.Markdown("## 📈 CSE Compass — Recommendation Engine for CSE Stocks")

    thread = gr.State(make_thread_id())

    message = gr.Textbox(show_label=False, placeholder="Your request to CSE Compass")

    company_input = gr.Textbox(
        label="Company Name",
        placeholder="e.g., John Keells Holdings / JKH / Commercial Bank ..."
    )
    intent_input = gr.Radio(["BUY", "SELL"], label="Action", value="BUY")
    buy_price_input = gr.Number(label="Bought Price (LKR)", visible=False, precision=2)

    intent_input.change(toggle_buy_price, inputs=intent_input, outputs=buy_price_input)

    submit_btn = gr.Button("Get Recommendation")
    output = gr.Textbox(label="Output", lines=12)

    submit_btn.click(
        process_message,
        inputs=[message, company_input, intent_input, buy_price_input, thread],
        outputs=output
    )

demo.launch()