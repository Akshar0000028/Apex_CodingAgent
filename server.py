"""
Multi-Agent Coding Assistant Backend  —  v3  (Deep Analysis)
=============================================================
Changes in v3:
  - SYSTEM_PROMPT upgraded: agent now MUST analyse the request deeply,
    plan which tools to chain, execute them, then synthesise a final answer.
  - All tools upgraded with richer, more thorough prompts.
  - on_tool_end now also sends a 'tool_done' SSE event so the UI can
    show a completion tick on each step badge.

Install:
    pip install fastapi uvicorn langchain langchain-nvidia-ai-endpoints \
                langchain-core langgraph python-dotenv

Set key:
    export NVIDIA_API_KEY="nvapi-your-key-here"

Run:
    uvicorn server:app --reload --port 8000
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="APEX — Multi-Agent Coding Assistant v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
NVIDIA_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

def _make_llm(temperature: float, max_tokens: int) -> ChatNVIDIA:
    return ChatNVIDIA(
        model=NVIDIA_MODEL,
        nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

# Shared LLM instances — created once at startup
_llm_precise  = _make_llm(temperature=0.1, max_tokens=3000)
_llm_balanced = _make_llm(temperature=0.2, max_tokens=3000)
_llm_explain  = _make_llm(temperature=0.3, max_tokens=3000)
_llm_agent    = _make_llm(temperature=0.2, max_tokens=3000)


# ── Tools — each with a deep, thorough prompt ─────────────────────────────────

@tool
def generate_code(task: str) -> str:
    """
    Generate complete, production-ready code for a given task.
    Before writing, analyse: language choice, architecture, edge cases,
    error handling strategy, and security considerations.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior software architect and engineer with 15+ years of experience.\n\n"
            "BEFORE writing any code, reason step-by-step:\n"
            "1. Understand the full requirements and constraints.\n"
            "2. Choose the right data structures, patterns, and architecture.\n"
            "3. Identify all edge cases, error conditions, and security concerns.\n"
            "4. Plan the module/function structure.\n\n"
            "THEN produce:\n"
            "- Complete, runnable code (no placeholders like '# TODO').\n"
            "- Type hints on every function signature.\n"
            "- Docstrings (Google style) for every function/class.\n"
            "- Inline comments explaining non-obvious logic.\n"
            "- Proper exception handling with meaningful error messages.\n"
            "- Input validation where appropriate.\n\n"
            "End with a SHORT section titled '## Design Decisions' explaining "
            "the key choices made and why."
        )),
        HumanMessage(content=f"Task: {task}")
    ])
    return (prompt | _llm_precise | StrOutputParser()).invoke({})


@tool
def review_code(code: str) -> str:
    """
    Perform a thorough multi-pass code review covering correctness, security,
    performance, style, and maintainability.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a principal engineer conducting a rigorous code review.\n\n"
            "Perform FIVE distinct review passes:\n\n"
            "PASS 1 — CORRECTNESS\n"
            "  Check logic errors, off-by-one errors, null/None dereferences, "
            "  race conditions, incorrect assumptions.\n\n"
            "PASS 2 — SECURITY\n"
            "  Check injection risks (SQL, command, path traversal), "
            "  authentication/authorisation flaws, secrets in code, "
            "  insecure defaults, unvalidated inputs.\n\n"
            "PASS 3 — PERFORMANCE\n"
            "  Check algorithmic complexity, unnecessary loops, missing indexes, "
            "  memory leaks, blocking I/O in async context, N+1 query patterns.\n\n"
            "PASS 4 — MAINTAINABILITY\n"
            "  Check naming clarity, function length, single-responsibility, "
            "  magic numbers, missing docstrings, code duplication.\n\n"
            "PASS 5 — BEST PRACTICES\n"
            "  Check language idioms, error handling patterns, test coverage gaps, "
            "  dependency management, logging.\n\n"
            "For EACH finding use the format:\n"
            "  [SEVERITY: CRITICAL/HIGH/MEDIUM/LOW] Line ~N: <issue>\n"
            "  Why: <explanation>\n"
            "  Fix: <concrete code or action>\n\n"
            "End with a SUMMARY SCORECARD (0-10) for each dimension and an overall grade."
        )),
        HumanMessage(content=f"Review this code:\n\n```\n{code}\n```")
    ])
    return (prompt | _llm_precise | StrOutputParser()).invoke({})


@tool
def debug_code(code_and_error: str) -> str:
    """
    Deeply diagnose a bug: identify root cause, explain the failure chain,
    provide a corrected version, and suggest tests to prevent regression.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert debugger and systems analyst.\n\n"
            "Follow this structured debugging methodology:\n\n"
            "STEP 1 — SYMPTOM ANALYSIS\n"
            "  Describe exactly what the error or unexpected behaviour means.\n\n"
            "STEP 2 — ROOT CAUSE IDENTIFICATION\n"
            "  Trace the exact execution path that leads to the failure.\n"
            "  Point to the specific line(s) responsible.\n\n"
            "STEP 3 — WHY IT HAPPENS\n"
            "  Explain the underlying language/runtime/library behaviour "
            "  that causes this. Be precise.\n\n"
            "STEP 4 — CORRECTED CODE\n"
            "  Provide the full corrected snippet with all fixes applied.\n"
            "  Mark every change with a comment: # FIX: <reason>\n\n"
            "STEP 5 — RELATED RISKS\n"
            "  List any similar bugs that may exist elsewhere in the code.\n\n"
            "STEP 6 — REGRESSION TESTS\n"
            "  Write 2-3 pytest test cases that would catch this bug "
            "  and verify the fix."
        )),
        HumanMessage(content=f"Debug this:\n\n{code_and_error}")
    ])
    return (prompt | _llm_precise | StrOutputParser()).invoke({})


@tool
def explain_code(code: str) -> str:
    """
    Explain code at multiple levels: high-level purpose, line-by-line mechanics,
    patterns used, and potential gotchas.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert coding mentor who can explain any code to any audience.\n\n"
            "Structure your explanation in FOUR layers:\n\n"
            "LAYER 1 — THE BIG PICTURE (2-3 sentences)\n"
            "  What does this code do and why would someone write it?\n\n"
            "LAYER 2 — HOW IT WORKS (step-by-step walkthrough)\n"
            "  Walk through the logic flow. For each key section explain:\n"
            "  - What it does\n"
            "  - Why it does it that way\n"
            "  - Any non-obvious language features used\n\n"
            "LAYER 3 — PATTERNS & CONCEPTS\n"
            "  Identify design patterns, algorithms, or paradigms used.\n\n"
            "LAYER 4 — GOTCHAS & EDGE CASES\n"
            "  What assumptions does the code make? What inputs would break it?\n"
            "  What would a developer commonly misunderstand about this code?"
        )),
        HumanMessage(content=f"Explain this code:\n\n```\n{code}\n```")
    ])
    return (prompt | _llm_explain | StrOutputParser()).invoke({})


@tool
def write_docs(code: str) -> str:
    """
    Generate complete, professional documentation: module overview,
    Google-style docstrings, usage examples, and a README section.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a senior technical writer specialising in developer documentation.\n\n"
            "Produce ALL of the following:\n\n"
            "1. MODULE / FILE DOCSTRING\n"
            "   Brief description, author placeholder, version, dependencies.\n\n"
            "2. FUNCTION / CLASS DOCSTRINGS (Google style) for EVERY public symbol:\n"
            "   - One-line summary\n"
            "   - Extended description if needed\n"
            "   - Args: type and description for each parameter\n"
            "   - Returns: type and description\n"
            "   - Raises: exceptions that may be raised\n"
            "   - Example: a concrete usage example\n\n"
            "3. INLINE COMMENTS for any logic that isn't self-evident.\n\n"
            "4. README SECTION (markdown) with:\n"
            "   - ## Overview\n"
            "   - ## Installation / Requirements\n"
            "   - ## Quick Start (with code snippet)\n"
            "   - ## API Reference summary table\n"
            "   - ## Notes / Limitations"
        )),
        HumanMessage(content=f"Write documentation for:\n\n```\n{code}\n```")
    ])
    return (prompt | _llm_balanced | StrOutputParser()).invoke({})


@tool
def optimize_code(code: str) -> str:
    """
    Optimise code for speed, memory, and readability.
    Profile hotspots, apply algorithmic improvements, and explain every change.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a performance engineering expert.\n\n"
            "Analyse and optimise in THREE phases:\n\n"
            "PHASE 1 — PROFILING ANALYSIS\n"
            "  Without running the code, identify:\n"
            "  - Time complexity of the current implementation (Big-O)\n"
            "  - Memory usage patterns and potential leaks\n"
            "  - Hotspots: which lines/loops/calls are likely slowest\n\n"
            "PHASE 2 — OPTIMISED CODE\n"
            "  Rewrite with all optimisations applied.\n"
            "  Mark every change: # OPT: <what changed and why>\n"
            "  Maintain identical external behaviour.\n\n"
            "PHASE 3 — OPTIMISATION REPORT\n"
            "  Table: | Change | Before complexity | After complexity | Expected speedup |\n"
            "  End with overall estimated improvement and trade-offs."
        )),
        HumanMessage(content=f"Optimise this code:\n\n```\n{code}\n```")
    ])
    return (prompt | _llm_precise | StrOutputParser()).invoke({})


# ── Tools list ────────────────────────────────────────────────────────────────
TOOLS = [generate_code, review_code, debug_code, explain_code, write_docs, optimize_code]


# ── System Prompt — forces deep analysis before every answer ──────────────────
SYSTEM_PROMPT = """You are APEX, an elite Multi-Agent Coding Assistant powered by NVIDIA AI.

══════════════════════════════════════════════════════════
MANDATORY ANALYSIS PROTOCOL — follow for EVERY request
══════════════════════════════════════════════════════════

STEP A — DEEP UNDERSTANDING
  Before doing anything else, fully understand the user's request:
  • What is the exact problem or goal?
  • What programming language / framework is involved?
  • What constraints exist (performance, style, existing codebase)?
  • What is the user's likely context?

STEP B — TOOL SELECTION PLAN
  Decide which tools to call and in what order.
  Available tools:
    generate_code  → write new code from scratch
    review_code    → audit for bugs, security, style (5-pass review)
    debug_code     → diagnose and fix errors (6-step methodology)
    explain_code   → explain what code does (4-layer explanation)
    write_docs     → create complete documentation
    optimize_code  → improve speed / memory / clarity (3-phase analysis)

  Chain tools when the task benefits from it:
    "write a fast, documented auth module"
      → generate_code → optimize_code → write_docs
    "fix my code and explain what was wrong"
      → debug_code → explain_code
    "review and optimise this function"
      → review_code → optimize_code
    "write production code with full docs"
      → generate_code → review_code → write_docs

STEP C — EXECUTE
  Call the planned tools in order. Pass complete, detailed context to each tool.
  Do NOT truncate code when passing it between tools.

STEP D — SYNTHESISE FINAL ANSWER
  After all tools complete, write a clear, structured final answer:
  • Start with a one-paragraph summary of what was done and why.
  • Present each tool's key findings in a logical, readable way.
  • Add expert commentary that connects the outputs.
  • End with concrete next steps or recommendations for the user.

══════════════════════════════════════════════════════════
QUALITY STANDARDS
══════════════════════════════════════════════════════════
• Never give a shallow one-tool answer when chaining adds genuine value.
• Never skip the analysis — always reason before acting.
• Be specific, technical, and thorough in every response.
• Use markdown headings and code blocks to structure output clearly.
• Treat every request as if it is going to production code.
"""


# ── Agent — built ONCE at startup ─────────────────────────────────────────────
print("⚙  Building APEX agent (once at startup)…")
_agent = create_react_agent(
    model=_llm_agent,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)
print("✓  APEX agent ready.")


# ── Request/Response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    agent_steps: list
    model_used: str

class DirectToolRequest(BaseModel):
    tool_name: str
    input_text: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_key():
    if not os.environ.get("NVIDIA_API_KEY"):
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY not set")

TOOL_MAP = {
    "generate": generate_code,
    "review":   review_code,
    "debug":    debug_code,
    "explain":  explain_code,
    "docs":     write_docs,
    "optimize": optimize_code,
}

LLM_MAP = {
    "generate": _llm_precise,
    "review":   _llm_precise,
    "debug":    _llm_precise,
    "explain":  _llm_explain,
    "docs":     _llm_balanced,
    "optimize": _llm_precise,
}

SYSTEM_MAP = {
    "generate": (
        "You are a senior software architect. Before writing any code, analyse requirements, "
        "choose the right architecture and patterns, then produce complete, production-ready "
        "code with type hints, docstrings, error handling, and a Design Decisions section."
    ),
    "review": (
        "You are a principal engineer. Perform a 5-pass review: CORRECTNESS, SECURITY, "
        "PERFORMANCE, MAINTAINABILITY, BEST PRACTICES. For each finding state severity, "
        "line, why it's a problem, and the fix. End with a scorecard and overall grade."
    ),
    "debug": (
        "You are an expert debugger. Follow 6 steps: symptom analysis, root cause, "
        "why it happens, corrected code (# FIX comments), related risks, regression tests."
    ),
    "explain": (
        "You are a coding mentor. Explain in 4 layers: big picture, step-by-step walkthrough, "
        "patterns & concepts, and gotchas & edge cases."
    ),
    "docs": (
        "You are a technical writer. Produce: module docstring, Google-style docstrings "
        "for every public symbol, inline comments, and a full README section in markdown."
    ),
    "optimize": (
        "You are a performance engineer. Work in 3 phases: profiling analysis (Big-O, hotspots), "
        "optimised code (# OPT comments), and an optimisation report table."
    ),
}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "model": NVIDIA_MODEL, "tools": len(TOOLS)}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "nvidia_key_set": bool(os.environ.get("NVIDIA_API_KEY")),
        "model": NVIDIA_MODEL,
        "tools": [t.name for t in TOOLS],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Orchestrator — full response, no streaming."""
    _require_key()
    try:
        result   = _agent.invoke({"messages": [HumanMessage(content=req.message)]})
        messages = result.get("messages", [])
        steps: list = []
        final_response = ""

        for msg in messages:
            t = type(msg).__name__
            if t == "AIMessage" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    steps.append({"tool": tc.get("name","unknown"),
                                   "input": str(tc.get("args",""))[:200],
                                   "output": ""})
            elif t == "ToolMessage":
                for step in reversed(steps):
                    if step["output"] == "":
                        step["output"] = str(msg.content)[:300]; break
            elif t == "AIMessage":
                c = msg.content
                final_response = (" ".join(b.get("text","") for b in c if isinstance(b,dict))
                                   if isinstance(c, list) else str(c))

        return ChatResponse(response=final_response or "No response generated.",
                            agent_steps=steps, model_used=NVIDIA_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Orchestrator — SSE streaming with step + token events."""
    _require_key()

    async def event_generator():
        steps: list = []
        try:
            async for event in _agent.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_tool_start":
                    step = {"tool": name,
                             "input": str(event.get("data",{}).get("input",""))[:200],
                             "output": ""}
                    steps.append(step)
                    yield f"data: {json.dumps({'type':'step','tool':name,'input':step['input']})}\n\n"

                elif kind == "on_tool_end":
                    for step in reversed(steps):
                        if step["output"] == "":
                            step["output"] = str(event.get("data",{}).get("output",""))[:300]
                            break
                    # Notify UI the tool finished so it can tick the badge
                    yield f"data: {json.dumps({'type':'tool_done','tool':name})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        c = chunk.content
                        text = ("".join(b.get("text","") for b in c if isinstance(b,dict))
                                if isinstance(c, list) else (str(c) if c else ""))
                        if text:
                            yield f"data: {json.dumps({'type':'token','text':text})}\n\n"

            yield f"data: {json.dumps({'type':'done','steps':steps})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/tool")
def call_tool_directly(req: DirectToolRequest):
    """Direct tool call — no orchestrator."""
    _require_key()
    selected = TOOL_MAP.get(req.tool_name)
    if not selected:
        raise HTTPException(status_code=400,
                            detail=f"Unknown tool. Choose from: {list(TOOL_MAP)}")
    try:
        return {"response": selected.invoke(req.input_text),
                "tool": req.tool_name, "model": NVIDIA_MODEL}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/stream")
async def call_tool_stream(req: DirectToolRequest):
    """Direct tool call — SSE streaming."""
    _require_key()
    if req.tool_name not in LLM_MAP:
        raise HTTPException(status_code=400,
                            detail=f"Unknown tool. Choose from: {list(TOOL_MAP)}")

    llm    = LLM_MAP[req.tool_name]
    system = SYSTEM_MAP[req.tool_name]

    async def event_generator():
        try:
            msgs = [SystemMessage(content=system), HumanMessage(content=req.input_text)]
            async for chunk in llm.astream(msgs):
                c = chunk.content
                text = ("".join(b.get("text","") for b in c if isinstance(b,dict))
                        if isinstance(c, list) else (str(c) if c else ""))
                if text:
                    yield f"data: {json.dumps({'type':'token','text':text})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})