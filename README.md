# Autonomous AI Agent with Memory & Tool Use

> A multi-tool autonomous agent built with LangGraph's ReAct loop, persistent FAISS memory, and a live streaming Gradio interface — deployed via FastAPI on Google Colab.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini_API-LLM-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Memory-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project implements a fully autonomous AI agent capable of planning, reasoning, and executing multi-step tasks without human intervention at each step. The agent uses the **ReAct (Reason + Act)** framework via LangGraph to iteratively think through a problem, select the right tool, observe the result, and loop until the task is complete.

Unlike a standard chatbot or RAG pipeline, this agent:
- **Decides** which tool to call based on the task
- **Remembers** past interactions across sessions using a persistent FAISS vector store
- **Executes** real actions — web searches, Python code, factual lookups
- **Streams** its reasoning chain live to a Gradio interface

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Gradio Interface                  │
│         (Live reasoning chain display)              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              FastAPI Server (SSE stream)             │
└────────────────────┬────────────────────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │       LangGraph Agent Core      │
     │     Plan → Act → Observe loop  │
     └───────┬───────────┬────────────┘
             │           │
    ┌────────▼──┐   ┌────▼──────────┐
    │ FAISS     │   │  Gemini API   │
    │ Memory    │   │  (reasoning)  │
    └───────────┘   └───────────────┘
             │
     ┌───────▼──────────────────────┐
     │         Tool Router          │
     └──────┬───────────┬───────────┘
            │           │           │
     ┌──────▼──┐ ┌──────▼──┐ ┌─────▼──────┐
     │  Tavily │ │ Python  │ │ Wikipedia  │
     │  Search │ │  REPL   │ │  Lookup    │
     └─────────┘ └─────────┘ └────────────┘
```

**Data flow:**
1. User sends a query via Gradio
2. FastAPI receives the request and starts an SSE stream
3. LangGraph agent plans the task using Gemini API
4. Agent checks FAISS memory for relevant past context
5. Tool router selects and calls the appropriate tool
6. Agent observes the result, updates state, and loops if needed
7. Final answer streamed back token-by-token to Gradio

---

## Results

| Metric | Value |
|---|---|
| Task completion rate | 85%+ across 20 benchmark tasks |
| Tool selection accuracy | 90% (correct tool chosen on first attempt) |
| Average reasoning steps | 3.2 steps per task |
| Average response latency | < 4s end-to-end on Colab free GPU |
| Memory retrieval precision | Top-3 relevant memories per query |

**Benchmark task breakdown (20 tasks):**

| Category | Tasks | Completion |
|---|---|---|
| Research & summarisation | 8 | 87% |
| Python code generation & execution | 6 | 83% |
| Factual Q&A (multi-hop) | 6 | 100% |

---

## Key Features

- **ReAct reasoning loop** — agent explicitly plans before acting; each step is logged as Thought / Action / Observation
- **Persistent vector memory** — FAISS index saved to Google Drive between Colab sessions; agent recalls relevant past interactions
- **3 callable tools** — Tavily web search (live internet), Python REPL (code execution), Wikipedia (factual lookup)
- **Streaming FastAPI endpoint** — Server-Sent Events (SSE) stream each reasoning token as it is generated
- **Live Gradio interface** — reasoning chain scrolls in real time; users can see every thought, tool call, and observation
- **3-variant prompt ablation** — system prompt variants compared on planning quality and tool call efficiency

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (ReAct state machine) |
| LLM | Google Gemini API via `langchain-google-genai` |
| Memory | FAISS + `sentence-transformers` embeddings |
| Tools | Tavily Search, Python REPL, WikipediaQueryRun |
| API server | FastAPI + Uvicorn (SSE streaming) |
| Interface | Gradio |
| Environment | Google Colab (free tier) + Google Drive persistence |
| Evaluation | Pandas, Matplotlib, custom rubric |

---

## Project Structure

```
autonomous-ai-agent/
├── notebooks/
│   ├── 01_tool_testing.ipynb        # Week 1 — unit tests for all 3 tools
│   ├── 02_agent_core.ipynb          # Week 2 — LangGraph state machine + memory
│   ├── 03_evaluation.ipynb          # Week 3 — benchmark suite + results
│   └── 04_deployment.ipynb          # Week 4 — FastAPI + Gradio deployment
├── agent/
│   ├── graph.py                     # LangGraph state machine definition
│   ├── memory.py                    # FAISS memory store (load, save, retrieve)
│   ├── tools.py                     # Tool definitions (Tavily, REPL, Wikipedia)
│   └── prompts.py                   # System prompt variants
├── api/
│   ├── main.py                      # FastAPI app + SSE streaming endpoint
│   └── schemas.py                   # Request/response models
├── evaluation/
│   ├── benchmark_tasks.json         # 20 benchmark tasks with expected tool + steps
│   └── results.csv                  # Full results table (20 tasks × 4 metrics)
├── findings.md                      # 1-page evaluation findings + failure analysis
├── requirements.txt
└── README.md
```

---

## Quickstart (Google Colab)

### 1. Clone the repo

```bash
!git clone https://github.com/YOUR_USERNAME/autonomous-ai-agent.git
%cd autonomous-ai-agent
```

### 2. Install dependencies

```bash
!pip install -r requirements.txt
```

### 3. Set your API keys

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
```

> Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com) and a free Tavily key at [tavily.com](https://tavily.com).

### 4. Mount Google Drive (for persistent memory)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Run the agent

```python
from agent.graph import build_agent
from agent.memory import load_memory

memory = load_memory("/content/drive/My Drive/agent_memory.index")
agent = build_agent(memory=memory)

result = agent.invoke({"input": "Research the latest developments in multimodal AI and write a Python script to visualise the trend"})
print(result["output"])
```

### 6. Launch the Gradio interface

```bash
!python api/main.py
```

The terminal will print a public URL — open it to see the live reasoning chain.

---

## Example Agent Run

```
User: Research transformer attention mechanisms and write a Python function to compute scaled dot-product attention.

Thought: I need to first gather information about transformer attention, then write and test code.

Action: tavily_search("transformer scaled dot-product attention mechanism 2024")
Observation: [Retrieved 3 articles on attention mechanisms from ArXiv and Distill.pub]

Thought: I now have enough context. Let me write the Python function.

Action: python_repl("""
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(weights, V), weights
""")
Observation: Function defined. No errors.

Thought: Code runs correctly. I can now provide the final answer with explanation.

Final Answer: [Explanation + working code returned to user]
```

---

## Evaluation Methodology

**20 benchmark tasks** were designed across 3 categories. Each task was scored on 4 metrics:

- **Completion** — did the agent produce a correct, useful final answer? (binary)
- **Tool accuracy** — was the correct tool selected on the first attempt? (binary)
- **Reasoning steps** — how many Plan → Act → Observe cycles were needed?
- **Latency** — wall-clock time from query to final answer (seconds)

**3 system prompt variants** were compared:
- Variant A: minimal instruction (baseline)
- Variant B: explicit step-by-step planning instruction
- Variant C: chain-of-thought with tool selection guidance

**Variant C achieved the highest task completion (85%+) and lowest average step count (3.2)**, showing that structured tool guidance significantly improves planning efficiency.

Full results: [`evaluation/results.csv`](evaluation/results.csv) — Findings: [`findings.md`](findings.md)

---

## Key Design Decisions

**Why LangGraph over bare LangChain agents?**
LangGraph gives explicit control over the agent's state machine — you can inspect, pause, or modify the graph at any node. This made debugging the ReAct loop far easier and is the direction the LangChain ecosystem is moving for production agents.

**Why FAISS over a hosted vector DB?**
FAISS runs entirely on CPU with no external service — zero cost, zero network latency, and fully reproducible. For a Colab environment it is the right choice. The index is persisted to Google Drive so memory survives session resets.

**Why Tavily over SerpAPI?**
Tavily is purpose-built for LLM agent use — it returns clean, structured snippets rather than raw HTML, which significantly reduces token waste in the context window.

**Why TinyLlama / Gemini API rather than a local 7B model?**
Free Colab GPU (T4, 15GB) cannot reliably run a 7B model alongside FAISS, FastAPI, and Gradio simultaneously. Gemini API keeps the compute off-device while still exposing a fully capable LLM, and the architecture remains identical when swapping to any other LLM provider.

---

## What I Would Do Next

- Add an **email/calendar tool** (Gmail API) to make the agent capable of real-world task execution
- Implement **multi-agent orchestration** — a planner agent spawning specialist sub-agents per task category
- Add **human-in-the-loop confirmation** before the Python REPL executes any code
- Migrate memory to **ChromaDB** for production-grade filtering and metadata search
- Publish benchmark results as a short technical report on arXiv

---

## Requirements

```
langchain>=0.2.0
langgraph>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.2.0
faiss-cpu>=1.7.4
sentence-transformers>=2.7.0
tavily-python>=0.3.0
fastapi>=0.111.0
uvicorn>=0.30.0
gradio>=4.36.0
pandas>=2.0.0
matplotlib>=3.8.0
pyngrok>=7.0.0
python-dotenv>=1.0.0
```

Install all: `pip install -r requirements.txt`

---

## License

MIT — free to use, modify, and build on. Attribution appreciated.

---




