"""
Gradio chat interface for the ReAct agent.

Run:
    python gradio_app/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from config import settings

PHASE_ICON = {"reason": "🧠", "act": "⚡", "load_memory": "🗄️", "save_memory": "💾"}

CSS = """
.container { max-width: 1100px !important; margin: auto; }
.trace-box { font-family: monospace; font-size: 0.85rem; }
footer { display: none !important; }
"""


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_task(task: str, history: list):
    """Blocking agent call — updates chatbot with trace + answer."""
    if not task.strip():
        return history, ""

    history = history + [(task, "⏳ Thinking…")]
    yield history, ""

    try:
        from agent.graph import run_agent
        t0 = time.perf_counter()
        state = run_agent(task)
        elapsed = time.perf_counter() - t0

        # Build readable trace
        lines = ["### 🔍 Reasoning Trace\n"]
        for step in state.get("trace", []):
            icon = PHASE_ICON.get(step.get("action", ""), "•")
            lines.append(
                f"**Step {step['iteration']}** · `{step['action']}` "
                f"({step.get('latency_ms', 0)}ms)\n"
                f"> 💭 {step['thought'][:250]}\n"
            )
            result = step.get("result", "")
            if result:
                snippet = result[:400] + ("…" if len(result) > 400 else "")
                lines.append(f"> 📋 `{snippet}`\n")

        answer = state.get("final_answer") or "Agent could not produce a final answer."
        lines += [
            "---",
            f"### ✅ Final Answer\n{answer}",
            f"\n*{state.get('iteration', 0)} steps · {elapsed:.2f}s*",
        ]

        history[-1] = (task, "\n".join(lines))

    except Exception as exc:
        history[-1] = (task, f"❌ **Error:** {exc}")

    yield history, ""


def search_memory(query: str) -> str:
    if not query.strip():
        return "Enter a query above."
    try:
        from memory.store import MemoryStore
        results = MemoryStore().search(query, top_k=5)
        if not results:
            return "No relevant memories found."
        lines = [f"**Results for:** `{query}`\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. *(score {r['score']:.2f})* {r['text'][:200]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Memory search error: {e}"


def get_memory_stats() -> str:
    try:
        from memory.store import MemoryStore
        return f"**{MemoryStore().size}** memories stored"
    except Exception:
        return "Memory unavailable"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Autonomous ReAct Agent",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 🤖 Autonomous ReAct Agent
        **LangGraph · Gemini 1.5 Pro · FAISS Memory · Web Search · Python REPL · Wikipedia**
        """
    )

    with gr.Row():
        # ── Left: chat ────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Agent",
                height=540,
                bubble_full_width=False,
                show_copy_button=True,
            )
            with gr.Row():
                task_box = gr.Textbox(
                    placeholder="Ask anything — the agent will reason step by step…",
                    label="",
                    lines=2,
                    scale=5,
                )
                run_btn = gr.Button("▶ Run", variant="primary", scale=1)

            gr.Examples(
                label="Try these",
                examples=[
                    ["What is the 17th Fibonacci number? Show your Python calculation."],
                    ["Search the web: latest breakthroughs in nuclear fusion 2024."],
                    ["Who invented the World Wide Web and when? Use Wikipedia."],
                    ["Write Python to find all primes under 50 and sum them."],
                    ["Compare the populations of Tokyo and New York City."],
                ],
                inputs=task_box,
            )

        # ── Right: memory panel ───────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Memory Search")
            mem_query = gr.Textbox(label="Search past sessions", placeholder="e.g. Fibonacci")
            mem_btn = gr.Button("Search", variant="secondary")
            mem_out = gr.Markdown(value=get_memory_stats())

            gr.Markdown("---")
            gr.Markdown(
                f"**Model:** `{settings.gemini_model}`  \n"
                f"**Max steps:** {settings.max_iterations}  \n"
                f"**Tools:** web_search · python_repl · wikipedia"
            )

    # ── Events ────────────────────────────────────────────────────────
    run_btn.click(run_task, [task_box, chatbot], [chatbot, task_box])
    task_box.submit(run_task, [task_box, chatbot], [chatbot, task_box])
    mem_btn.click(search_memory, mem_query, mem_out)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=settings.gradio_share,
        show_error=True,
    )
