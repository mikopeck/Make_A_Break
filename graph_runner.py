# graph_runner.py
from .langgraph_setup import jailbreak_graph, JailbreakAttemptState # Assuming graph is compiled here
import json
import os
from datetime import datetime

RESULTS_LOG_FILE = os.path.join("results", "jailbreak_log.jsonl")
os.makedirs("results", exist_ok=True)

def run_single_jailbreak_attempt(
    task: dict, 
    strategy: dict, 
    target_model_name: str, 
    judge_model_name: str
) -> dict:
    """
    Runs a single task-strategy pair through the jailbreak graph.
    """
    initial_state = JailbreakAttemptState(
        task=task,
        strategy=strategy,
        target_model_name=target_model_name,
        judge_model_name=judge_model_name,
        jailbreak_prompt=None,
        target_llm_response=None,
        final_verdict=None,
        verdict_reasoning=None,
        error_message=None,
        log=[]
    )

    final_state = jailbreak_graph.invoke(initial_state)
    
    # Log the result
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task.get("id"),
        "task_prompt": task.get("prompt"),
        "strategy_id": strategy.get("id"),
        "strategy_name": strategy.get("name"),
        "target_model_name": target_model_name,
        "judge_model_name": judge_model_name,
        "jailbreak_prompt": final_state.get("jailbreak_prompt"),
        "target_llm_response": final_state.get("target_llm_response"),
        "final_verdict": final_state.get("final_verdict"),
        "verdict_reasoning": final_state.get("verdict_reasoning"),
        "error_message": final_state.get("error_message"),
        "detailed_log": final_state.get("log")
    }
    
    with open(RESULTS_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return log_entry