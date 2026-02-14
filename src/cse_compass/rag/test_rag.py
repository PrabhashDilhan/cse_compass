from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (works regardless of cwd when run as module)
_project_root = Path(__file__).resolve().parents[3]
load_dotenv(_project_root / ".env")

from cse_compass.rag.retriever import retrieve
docs = retrieve(
    query="debt maturity and interest rate risk",
    filters={"ticker": "DFCC.N0000"},
    k=5,
)
for d in docs:
    print(d.metadata, d.page_content[:200])
