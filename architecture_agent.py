import json
from typing import Dict, Any, List, TypedDict, Annotated

import httpx
from langchain_openai import ChatOpenAI
import config
import logging
import traceback
from openai import InternalServerError
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Existing diagram generator (used as a tool)
from diagram_generator import generate_graphviz_diagram

# NEW: LangChain tools + agent
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ===========================
# AZURE OPENAI CLIENT
# ===========================

http_client = httpx.Client(verify=False)

client = ChatOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    base_url=config.AZURE_OPENAI_ENDPOINT,
    model=config.OPENAI_MODEL,
    http_client=http_client,
    temperature=0.1,
)

# ===========================
# Load templates
# ===========================

with open(config.TEMPLATES_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_DATA = json.load(f)


# ======================================================
# Build prompt for architecture design + refinement
# ======================================================

def build_prompt_messages(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    template_summaries = [
        {"id": p["id"], "name": p["name"], "description": p["description"]}
        for p in TEMPLATE_DATA.get("patterns", [])
    ]
    templates_str = json.dumps(template_summaries, indent=2)

    system_content = (
        "You are an Architecture Design Assistant for IT systems. "
        "Your job is to take high-level requirements and propose a system architecture.\n\n"
        "You have access to a small library of architecture patterns. "
        "Each pattern has an id, name, and description. Use them as reusable reference designs.\n\n"
        "Return ONLY JSON (no markdown outside the JSON block, no extra text). "
        "The JSON MUST have this structure:\n"
        "{\n"
        "  \"summary\": \"An HTML-formatted architecture summary.\",\n"
        "  \"pattern_id\": \"id of the pattern you are closest to (or 'custom' if none fits)\",\n"
        "  \"components\": [\n"
        "    {\"id\": \"short_id\", \"label\": \"Readable name\", \"type\": \"e.g. web, app, db, cache, queue, mobile_client\"}\n"
        "  ],\n"
        "  \"connections\": [\n"
        "    {\"from\": \"component_id\", \"to\": \"component_id\", \"label\": \"protocol or purpose\"}\n"
        "  ]\n"
        "}\n"
        "IDs must be valid Graphviz node identifiers (letters, digits, underscores only). "
        "Use about 4–12 components to keep the diagram readable.\n\n"
        "IMPORTANT: The `summary` field MUST be valid HTML, not markdown. Use tags like:\n"
        "- <h3>Overview</h3>\n"
        "- <h3>Key Components</h3>\n"
        "- <h3>Data Flow</h3>\n"
        "- <h3>Scalability & Reliability</h3>\n"
        "Within each section, use <ul><li>.</li></ul> bullet lists.\n\n"
        "SUMMARY LENGTH RULES:\n"
        "- Keep the HTML formatting EXACTLY the same (h3 headings + bullet lists).\n"
        "- Keep all <ul><li>.</li></ul> bullet lists.\n"
        "- Make the summary concise: shorten each bullet point using brief, telegraphic text.\n"
        "- Keep the meaning but remove verbosity.\n"
        "- Target 40–60% of the usual summary length.\n\n"
        "REFINEMENT RULES:\n"
        "- If a previous architecture plan is provided, treat it as the BASELINE.\n"
        "- You MUST keep existing component IDs and labels as stable as possible.\n"
        "- Prefer to ADD components or connections rather than renaming or deleting.\n"
        "- Only change or remove existing components if the new requirements clearly conflict.\n"
        "- If a previous pattern_id is provided, keep the same pattern_id unless the user explicitly asks to change the pattern.\n"
    )

    user_parts: List[str] = []
    user_parts.append("Available architecture patterns:\n")
    user_parts.append(templates_str)

    if previous_arch_plan:
        user_parts.append("\n\nPrevious architecture plan (baseline):\n")
        user_parts.append(json.dumps(previous_arch_plan, indent=2))
        user_parts.append("\n\nUser refinement request:\n")
        user_parts.append(user_message)
    else:
        user_parts.append("\n\nFull user requirements:\n")
        user_parts.append(user_message)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "".join(user_parts)},
    ]
    return messages


# ======================================================
# Call architecture model
# ======================================================

def _call_model(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Low-level call to the architecture LLM. Used by the LangGraph node.
    """
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    messages = build_prompt_messages(user_message, previous_arch_plan)

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    full_prompt = system_content + "\n\n" + user_content

    try:
        llm_result = client.invoke(full_prompt)
        raw_text = getattr(llm_result, "content", str(llm_result))

        clean = raw_text.strip()
        if clean.startswith("```"):
            import re
            match = re.search(r"\{[\s\S]*\}", clean)
            if match:
                clean = match.group(0)

        try:
            arch_plan = json.loads(clean)
        except Exception:
            arch_plan = _fallback_architecture("Invalid JSON from model.")

    except InternalServerError as e:
        logger.error("Azure gateway returned 500: %s", e)
        raise RuntimeError("Azure gateway 500 — check logs.") from e

    except Exception as ex:
        logger.error("Azure OpenAI call failed: %s", ex)
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise RuntimeError("Connection failure to Azure OpenAI.") from ex

    arch_plan.setdefault("summary", "No summary provided.")
    arch_plan.setdefault("pattern_id", "unknown")
    arch_plan.setdefault("components", [])
    arch_plan.setdefault("connections", [])

    return arch_plan


def _fallback_architecture(reason: str) -> Dict[str, Any]:
    return {
        "summary": f"Fallback architecture: {reason}",
        "pattern_id": "fallback_three_tier",
        "components": [
            {"id": "client", "label": "Client", "type": "client"},
            {"id": "web", "label": "Web Server", "type": "web"},
            {"id": "app", "label": "App Server", "type": "app"},
            {"id": "db", "label": "Database", "type": "database"},
        ],
        "connections": [
            {"from": "client", "to": "web", "label": "HTTP"},
            {"from": "web", "to": "app", "label": "Internal HTTP"},
            {"from": "app", "to": "db", "label": "SQL"},
        ],
    }


# ======================================================
# LangGraph state + workflow 
# ======================================================

class ArchState(TypedDict):
    messages: Annotated[List[str], operator.add]
    arch_plan: Dict[str, Any]
    arch_history: Annotated[List[Dict[str, Any]], operator.add]


def _llm_node(state: ArchState) -> ArchState:
    """
    Single LangGraph node that:
    - reads latest requirements from state
    - looks at previous architecture (if any)
    - calls the architecture LLM
    - returns updated plan + history
    """
    msgs = state.get("messages") or []
    if not msgs:
        raise RuntimeError("No requirements text provided to LLM node.")

    latest_req = msgs[-1]

    hist = state.get("arch_history") or []
    previous_arch = hist[-1] if hist else None

    arch_plan = _call_model(latest_req, previous_arch)

    return {
        "messages": [],
        "arch_plan": arch_plan,
        "arch_history": [arch_plan],
    }


_graph_builder = StateGraph(ArchState)
_graph_builder.add_node("llm", _llm_node)
_graph_builder.set_entry_point("llm")
_graph_builder.add_edge("llm", END)

_checkpointer = MemorySaver()
_arch_graph = _graph_builder.compile(checkpointer=_checkpointer)


def call_llm_for_architecture(user_message: str, thread_id: str = "default") -> Dict[str, Any]:
    """
    Public API for architecture generation / refinement.
    This function is preserved so existing code can still call it directly.
    """
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    initial_state: ArchState = {
        "messages": [user_message],
        "arch_plan": {},
        "arch_history": [],
    }

    final = _arch_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    plan = final.get("arch_plan") or _fallback_architecture("Missing plan.")
    return plan


# ======================================================
# NFR VALIDATION
# ======================================================

def _call_nfr_model(requirements_text: str, arch_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls GPT-4o to produce NFR analysis.
    """
    system_prompt = (
        "You are an IT Architecture NFR reviewer. Your job is to analyze "
        "the architecture against standard NFR categories:\n"
        "- Performance\n- Scalability\n- Availability\n- Security\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"overall_risk\": \"Low|Medium|High\",\n"
        "  \"summary\": \"short text\",\n"
        "  \"issues\": [\n"
        "     {\"category\": \".\", \"severity\": \"Low|Medium|High\", "
        "      \"finding\": \".\", \"recommendation\": \".\"}\n"
        "  ]\n"
        "}\n"
    )

    user_prompt = (
        "Full Requirements:\n" + requirements_text + "\n\n"
        "Architecture Plan JSON:\n" + json.dumps(arch_plan, indent=2)
    )

    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        llm_result = client.invoke(full_prompt)
        text = getattr(llm_result, "content", str(llm_result)).strip()

        if text.startswith("```"):
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)

        result = json.loads(text)
        return result

    except Exception as e:
        logger.error("NFR validation failed: %s", e)
        return {
            "overall_risk": "Unknown",
            "summary": "NFR validation failed.",
            "issues": [],
        }


def validate_nfr_for_architecture(
    full_requirements_text: str,
    arch_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Public wrapper for NFR agent. Kept as-is so other callers can still use it directly.
    """
    if not arch_plan:
        return {
            "overall_risk": "Unknown",
            "summary": "No architecture plan to validate.",
            "issues": [],
        }

    return _call_nfr_model(full_requirements_text, arch_plan)


# ======================================================
# NFR NEEDS-DETECTION CLASSIFIER
# ======================================================

def _detect_nfr_need(requirements_text: str) -> bool:
    """
    Uses the LLM as a classifier to decide whether NFR validation is needed
    for the given requirements.

    Returns True if the user seems to care about NFRs (performance, latency,
    scalability, availability, security, compliance, etc.), otherwise False.
    """
    system_prompt = (
        "You are an assistant that decides whether a set of IT system requirements "
        "contains explicit or implicit non-functional requirements (NFRs).\n\n"
        "NFRs include things like: performance, latency, throughput, scalability, "
        "high availability, reliability, DR / RPO / RTO, security, compliance, "
        "data privacy, observability, monitoring, logging, resilience, fault tolerance, "
        "SLA, SLO, capacity, load handling, etc.\n\n"
        "Return ONLY valid JSON with this structure:\n"
        "{ \"needs_nfr_validation\": true or false }\n\n"
        "Answer true if the user mentions or clearly implies any of the NFR topics above. "
        "Otherwise, answer false."
    )

    user_prompt = "User requirements:\n" + requirements_text

    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        llm_result = client.invoke(full_prompt)
        text = getattr(llm_result, "content", str(llm_result)).strip()

        if text.startswith("```"):
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)

        data = json.loads(text)
        value = data.get("needs_nfr_validation")

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"

    except Exception as e:
        logger.error("NFR need detection failed: %s", e)

    # On error or unclear response, default to False to avoid surprising the user
    return False


# ======================================================
# AGENT TOOLS
# ======================================================

def _make_agent_tools(
    full_requirements_text: str,
    thread_id: str,
) -> List[Tool]:
    """
    Creates LangChain tools that wrap the existing functions:
    - call_llm_for_architecture
    - generate_graphviz_diagram
    - validate_nfr_for_architecture

    The outer scope carries full_requirements_text + thread_id so the tools
    don't need to be passed those explicitly by the LLM.
    """

    def _design_architecture(requirements: str | None = None) -> Dict[str, Any]:
        """
        Generate or refine an architecture plan from the requirements text.
        If requirements is omitted, use the full_requirements_text.
        """
        text = (requirements or full_requirements_text).strip()
        return call_llm_for_architecture(text, thread_id=thread_id)

    def _render_diagram(arch_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render a Graphviz diagram from the given architecture plan.
        Returns {\"image_url\": ..., \"dot\": ...}.
        """
        image_url, dot_source = generate_graphviz_diagram(arch_plan)
        return {"image_url": image_url, "dot": dot_source}

    def _validate_nfr(
        requirements: str | None = None,
        arch_plan: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Validate NFRs for the given architecture.
        If arch_plan is omitted, it will first call the architecture tool.
        """
        text = (requirements or full_requirements_text).strip()
        plan = arch_plan or _design_architecture(text)
        return validate_nfr_for_architecture(text, plan)

    tools: List[Tool] = [
        Tool(
            name="design_architecture",
            func=_design_architecture,
            description=(
                "Use this tool to generate or refine the system architecture plan "
                "as structured JSON (summary, pattern_id, components, connections) "
                "from the user's requirements."
            ),
        ),
        Tool(
            name="render_diagram",
            func=_render_diagram,
            description=(
                "Use this tool to generate an architecture diagram (SVG) from a JSON "
                "architecture plan. It returns a dict with 'image_url' and 'dot'."
            ),
        ),
        Tool(
            name="validate_nfr",
            func=_validate_nfr,
            description=(
                "Use this tool to run Non-Functional Requirements (NFR) analysis for "
                "performance, scalability, availability, security, etc., on the "
                "current architecture plan and requirements."
            ),
        ),
    ]
    return tools


def _agent_system_message() -> SystemMessage:
    """
    System message that instructs the agent how to use tools and what final
    JSON shape to return.
    """
    content = (
        "You are an Interactive IT Architecture Design Assistant with tools.\n\n"
        "TOOLS:\n"
        "1) design_architecture(requirements: str | None)\n"
        "   - Always use this to generate or refine the architecture JSON plan.\n"
        "   - It returns a dict with: summary, pattern_id, components, connections.\n\n"
        "2) render_diagram(arch_plan: dict)\n"
        "   - Use this after you have an architecture plan to produce a diagram.\n"
        "   - It returns: {\"image_url\": \"path/to/svg\", \"dot\": \"graphviz source\"}.\n\n"
        "3) validate_nfr(requirements: str | None, arch_plan: dict | None)\n"
        "   - Use this ONLY if the requirements mention or imply non-functional concerns:\n"
        "     performance, latency, throughput, scalability, high availability, DR, RPO/RTO,\n"
        "     security, compliance, privacy, observability, monitoring, capacity, SLAs, etc.\n"
        "   - It returns an NFR report JSON.\n\n"
        "DECISION LOGIC:\n"
        "- For every new or refined request, you MUST call design_architecture at least once.\n"
        "- You SHOULD call render_diagram after you have the final architecture plan so the user gets a diagram.\n"
        "- Call validate_nfr only when NFRs are clearly relevant. If in doubt, you may skip it.\n\n"
        "FINAL ANSWER FORMAT:\n"
        "After you finish using tools, your final reply MUST be ONLY a single JSON object with this shape:\n"
        "{\n"
        "  \"summary\": \"HTML summary string from the architecture plan.\",\n"
        "  \"pattern_id\": \"pattern id or 'custom' or 'unknown'\",\n"
        "  \"components\": [ ... ],\n"
        "  \"connections\": [ ... ],\n"
        "  \"image_url\": \"SVG path from render_diagram\" or null,\n"
        "  \"dot\": \"Graphviz DOT source\" or \"\",\n"
        "  \"nfr_report\": { ... } or null\n"
        "}\n\n"
        "RULES:\n"
        "- The 'summary', 'pattern_id', 'components', and 'connections' MUST come from the latest architecture plan.\n"
        "- 'image_url' and 'dot' MUST come from the most recent render_diagram call, if used.\n"
        "- If you did not call validate_nfr, set \"nfr_report\" to null.\n"
        "- DO NOT include any explanation text outside the JSON.\n"
    )
    return SystemMessage(content=content)

def _legacy_run_architecture_agent(
    full_requirements_text: str,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    Previous 'hybrid' behavior:
    - Always architecture
    - Always diagram
    - NFR based on classifier
    """
    arch_plan = call_llm_for_architecture(
        full_requirements_text,
        thread_id=thread_id,
    )

    image_url, dot_source = generate_graphviz_diagram(arch_plan)

    needs_nfr = _detect_nfr_need(full_requirements_text)

    if needs_nfr:
        nfr_report = validate_nfr_for_architecture(
            full_requirements_text,
            arch_plan,
        )
    else:
        nfr_report = None

    response_payload: Dict[str, Any] = {
        "summary": arch_plan.get("summary"),
        "pattern_id": arch_plan.get("pattern_id"),
        "components": arch_plan.get("components", []),
        "connections": arch_plan.get("connections", []),
        "image_url": image_url,
        "dot": dot_source,
        "nfr_report": nfr_report,
    }
    return response_payload

# ======================================================
# AGENTIC ORCHESTRATOR
# ======================================================
def run_architecture_agent(
    full_requirements_text: str,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    High-level agent-style orchestrator (agentic version).
    - Creates LangChain tools that wrap your existing LangGraph + diagram + NFR functions.
    - Uses an OpenAI-functions-style agent to decide which tools to call.
    - Agent must return a single JSON object with the fields expected by the UI.
    - On any error, falls back to the legacy orchestrator to avoid breaking behavior.
    """

    # Build tools bound to this specific conversation/requirements
    tools = _make_agent_tools(full_requirements_text, thread_id)

    system_msg = _agent_system_message()

    # Create the agent executor
    agent = initialize_agent(
        tools=tools,
        llm=client,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs={"system_message": system_msg},
    )

    try:
        # Let the agent decide which tools to call.
        # It will see the full requirements as input.
        raw_output = agent.run(full_requirements_text)

        text = str(raw_output).strip()
        if text.startswith("```"):
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)

        payload = json.loads(text)

        # Ensure required keys exist (for UI safety)
        payload.setdefault("summary", "")
        payload.setdefault("pattern_id", "unknown")
        payload.setdefault("components", [])
        payload.setdefault("connections", [])
        payload.setdefault("image_url", None)
        payload.setdefault("dot", "")
        payload.setdefault("nfr_report", None)

        return payload

    except Exception as e:
        logger.error("Agentic orchestration failed, falling back to legacy: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        return _legacy_run_architecture_agent(full_requirements_text, thread_id)
