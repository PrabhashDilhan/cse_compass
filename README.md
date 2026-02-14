# CSE Compass

An AI agentic application built with **LangGraph** that suggests Colombo Stock Exchange (CSE) stock trading recommendations. The system takes minimal user input—**company name** and **intent** (BUY/SELL)—and produces evidence-grounded analysis by orchestrating RAG, market data, web search, and report generation.

## Overview

CSE Compass uses LangGraph to create a multi-node agent pipeline that:

- Retrieves financial signals from ingested company reports (RAG)
- Fetches live market data via CSE MCP tools
- Searches the web for recent news, CSE updates, Sri Lanka macro, and global risks
- Synthesizes an analytical report with a clear recommendation
- Optionally sends the report via email

## User Inputs

| Input | Description |
|------|-------------|
| **Company name** | e.g. "John Keells Holdings" |
| **Intent** | `BUY` or `SELL` |

Optional: `buy_price` (for SELL intent, to compute approximate P/L).

## LangGraph Architecture

The application is structured as a LangGraph state machine with the following nodes:

```mermaid
flowchart TD
    START([START]) --> rag_summarizer_node
    rag_summarizer_node[RAG Summarizer Node] --> cse_mcp_node

    cse_mcp_node[CSE MCP Node] -->|tool calls| cse_mcp_tool
    cse_mcp_node -->|no tool calls| web_search_node
    cse_mcp_tool[CSE MCP Tool] --> cse_mcp_node

    web_search_node[Web Search Node] -->|tool calls| web_search_tool
    web_search_node -->|no tool calls| cse_compass_report_node
    web_search_tool[Web Search Tool] --> web_search_node

    cse_compass_report_node[CSE Compass Report Node] --> send_email_node
    send_email_node[Send Email Node] --> END([END])

    subgraph "LangGraph Nodes"
        rag_summarizer_node
        cse_mcp_node
        web_search_node
        cse_compass_report_node
        send_email_node
    end

    subgraph Tools
        cse_mcp_tool
        web_search_tool
    end
```

### Node Descriptions

| Node | Purpose |
|------|---------|
| **rag_summarizer_node** | Queries RAG over financial reports, extracts signals (revenue, profitability, debt, cash flow, risks), and produces a structured summary |
| **cse_mcp_node** | Calls CSE MCP tools for live market data (price, change, change %); may loop via CSE MCP Tool until data is retrieved |
| **CSE MCP Tool** | Executes CSE MCP market data tool calls |
| **web_search_node** | Plans and runs web searches for company news, CSE market updates, Sri Lanka macro, and global risks; may loop via Web Search Tool |
| **Web Search Tool** | Executes web search tool calls |
| **cse_compass_report_node** | Combines RAG summary, market data, and web results into a JSON report (confidence, decision, key points, risks, watchlist triggers, email draft) |
| **send_email_node** | Sends the report via email using the configured email tool |

## Setup

The `data/` directory is not included in the repo. Create it and add financial reports before running RAG ingestion.

### 1. Create directory structure

```bash
mkdir -p data/raw/cse
mkdir -p data/vectorstore/chroma_db
```

### 2. Add financial reports

Choose one of the following options:

**Option A: Raw PDFs**

Place PDF files under `data/raw/cse/` (or a subdirectory). PDFs are discovered recursively.

- **Filename format** (recommended): `{ticker}_{year}_{type}.pdf`  
  Examples: `JKH_2023_annual.pdf`, `DFCC_2024_q1.pdf`
- **Alternative**: Organize by folder, e.g. `data/raw/cse/JKH/2023/annual.pdf`

**Option B: Processed reports**

If you have pre-processed reports (text.md, tables.json, metadata.json), place them under:

```
data/raw/cse/processed_reports/
  └── {Company Name (TICKER)}/
      └── annual|quarterly|press_release/
          └── {report_id}/
              ├── metadata.json
              ├── text.md   (or combined.md)
              └── tables.json
```

Example: `data/raw/cse/processed_reports/John Keells Holdings (JKH.N0000)/annual/371_1723200115194.06.2024/`

### 3. Run RAG ingestion

**For PDFs:**

```bash
python3 -m cse_compass.ingestion.ingest_reports \
  --input data/raw/cse \
  --persist data/vectorstore/chroma_db
```

**For processed reports:**

```bash
python3 -m cse_compass.ingestion.ingest_processed_reports \
  --input data/raw/cse/processed_reports \
  --persist data/vectorstore/chroma_db
```

Optional: add `--cache-chunks data/processed/chunks` to cache chunk JSONL files.

---

## Run

```bash
python3 -m cse_compass.graphs.graph
```

Ensure `.env` contains your API keys (OpenAI, Serper, SendGrid, etc.) and that the CSE MCP server is available.
