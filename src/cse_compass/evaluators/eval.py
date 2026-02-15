import os
import uuid
import asyncio
from typing import Any, Dict, Optional

from langsmith import aevaluate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from cse_compass.graphs.graph import graph
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ---------- helpers ----------
def make_thread_id() -> str:
    return str(uuid.uuid4())


def example_to_state(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ONE LangSmith dataset example "inputs" into your graph State dict.

    Your dataset input looks like:
    {
      "messages": [
        {
          "type": "human",
          "intent": "BUY",
          "ticker": "RCL.N0000",
          "company": "ROYAL CERAMICS LANKA PLC",
          "content": "I want to buy ...",
          ...
        }
      ]
    }
    """
    msg0 = (inputs.get("messages") or [{}])[0]

    content = msg0.get("content", "")
    intent = msg0.get("intent", None)
    ticker = msg0.get("ticker", None)
    company = msg0.get("company", None)

    # Build the minimal state your graph expects.
    # Add any other required keys your nodes assume exist.
    return {
        "messages": [HumanMessage(content=content)],
        "company_name": company,
        "ticker_from_company": ticker,
        "intent": intent,
        "buy_price": None,
        "rag_search_results": "",
        "web_search_results": {},
        "web_search_done": False,
    }


async def target_app(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the function LangSmith will call per example.
    It must accept a dict and return a dict.

    We:
      1) map example inputs -> graph state
      2) run graph.ainvoke with a unique thread_id
      3) return the graph outputs (state)
    """
    state = example_to_state(inputs)
    thread_id = make_thread_id()

    # LangGraph checkpointers often look at configurable.thread_id.
    # LangSmith "threads" grouping uses metadata.thread_id (or session_id/conversation_id).
    # We'll set both.
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }

    out_state = await graph.ainvoke(state, config=config)
    return out_state

client = OpenAI()

class Similarity_Score(BaseModel):
    similarity_score: int = Field(description="Semantic similarity score between 1 and 10, where 1 means unrelated and 10 means identical.")

# NOTE: This is our evaluator
def compare_semantic_similarity(reference_outputs: dict, outputs: dict):
    reference_response = reference_outputs.get("messages")[-1].content
    run_response = outputs.get("messages")[-1].content
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {   
                "role": "system",
                "content": (
                    "You are a semantic similarity evaluator. Compare the meanings of two responses to a question, "
                    "Reference Response and New Response, where the reference is the correct answer, and we are trying to judge if the new response is similar. "
                    "Provide a score between 1 and 10, where 1 means completely unrelated, and 10 means identical in meaning."
                ),
            },
            {"role": "user", "content": f" Reference Response: {reference_response}\n Run Response: {run_response}"}
        ],
        response_format=Similarity_Score,
    )

    similarity_score = completion.choices[0].message.parsed
    return {"score": similarity_score.similarity_score, "key": "similarity"}


async def main() -> None:
    dataset_name = "CSE_COMPASS_GOLDEN_DATASET"
    experiment_prefix = "cse-compass_llm_eval"

    # Option A (simple): pass target function directly
    # Option B: wrap as RunnableLambda; both work.
    target = RunnableLambda(target_app)

    results = await aevaluate(
        target,
        data=dataset_name,
        evaluators=[compare_semantic_similarity],
        experiment_prefix=experiment_prefix,
    )

    # Print a tiny summary in console
    print("Eval done.")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
