# langgraph_setup.py
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

from .llm_interface import query_ollama_model
from .judge import evaluate_response_with_ai_judge
import base64 # For potential strategies that encode prompts

# --- State Definition ---
class JailbreakAttemptState(TypedDict):
    task: dict
    strategy: dict
    target_model_name: str
    judge_model_name: str
    
    jailbreak_prompt: Optional[str]
    target_llm_response: Optional[str]
    final_verdict: Optional[str]
    verdict_reasoning: Optional[str]
    error_message: Optional[str]
    
    # To potentially store a log of operations if needed within the graph run
    log: List[str]


# --- Node Functions ---
def prepare_jailbreak_prompt_node(state: JailbreakAttemptState) -> JailbreakAttemptState:
    task_prompt = state["task"]["prompt"]
    strategy_template = state["strategy"]["template"]
    
    # Example for a strategy that might require special handling like base64
    if state["strategy"].get("id") == "S4_base64_encode_request": # Fictional ID from earlier thought process
        encoded_task_prompt = base64.b64encode(task_prompt.encode()).decode()
        jailbreak_prompt = strategy_template.format(base64_encoded_task_prompt=encoded_task_prompt)
    else:
        jailbreak_prompt = strategy_template.format(task_prompt=task_prompt)
        
    state["jailbreak_prompt"] = jailbreak_prompt
    state["log"] = state.get("log", []) + [f"Prepared jailbreak prompt: {jailbreak_prompt[:100]}..."]
    return state

def query_target_llm_node(state: JailbreakAttemptState) -> JailbreakAttemptState:
    if state.get("error_message"): return state # Skip if previous error
    try:
        response = query_ollama_model(
            model_name=state["target_model_name"],
            prompt=state["jailbreak_prompt"]
        )
        state["target_llm_response"] = response
        state["log"] = state.get("log", []) + [f"Target LLM response: {response[:100]}..."]
    except Exception as e:
        state["error_message"] = f"Error querying target LLM: {str(e)}"
    return state

def evaluate_response_node(state: JailbreakAttemptState) -> JailbreakAttemptState:
    if state.get("error_message") or not state.get("target_llm_response"): return state
    try:
        verdict, reasoning = evaluate_response_with_ai_judge(
            original_task_prompt=state["task"]["prompt"],
            llm_response=state["target_llm_response"],
            judge_model_name=state["judge_model_name"]
        )
        state["final_verdict"] = verdict
        state["verdict_reasoning"] = reasoning
        state["log"] = state.get("log", []) + [f"Verdict: {verdict}. Reasoning: {reasoning[:100]}..."]
    except Exception as e:
        state["error_message"] = f"Error during evaluation: {str(e)}"
        state["final_verdict"] = "EVALUATION_ERROR"
        state["verdict_reasoning"] = str(e)
    return state

# --- Graph Construction ---
def build_jailbreak_graph():
    workflow = StateGraph(JailbreakAttemptState)

    workflow.add_node("prepare_prompt", prepare_jailbreak_prompt_node)
    workflow.add_node("query_target_llm", query_target_llm_node)
    workflow.add_node("evaluate_response", evaluate_response_node)

    workflow.set_entry_point("prepare_prompt")
    workflow.add_edge("prepare_prompt", "query_target_llm")
    workflow.add_edge("query_target_llm", "evaluate_response")
    workflow.add_edge("evaluate_response", END)
    
    return workflow.compile()

# To be used by graph_runner.py
jailbreak_graph = build_jailbreak_graph()