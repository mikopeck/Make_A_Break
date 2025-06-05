# comprehensive_runner.py
import itertools
import re
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END
import streamlit as st

from llm_interface import query_ollama_model
from graph_runner import run_single_jailbreak_attempt

# --- System Prompt for Combination Crafter ---
COMBINATION_CRAFTER_PROMPT = "..." # (No changes to this prompt)

class ComprehensiveRunState(TypedDict):
    task: Dict[str, Any]
    all_strategies: List[Dict[str, Any]]
    model_config: Dict[str, str]
    ui_monitor_placeholders: Optional[Dict[str, Any]] # For live monitoring
    probing_results: List[Dict[str, Any]]
    top_4_strategies: List[Dict[str, Any]]
    strategy_combinations: List[tuple]
    combined_prompts: List[Dict[str, str]] # Store prompt and combo name
    final_assault_results: List[Dict[str, Any]]
    run_log: List[str]
    was_stopped: bool # To track if the run was stopped by user

def probing_stage_node(state: ComprehensiveRunState) -> ComprehensiveRunState:
    log = ["**Stage 1: Probing Runs** - Testing each strategy 3 times..."]
    st.session_state['comp_log'] = log
    
    results = []
    for strategy in state["all_strategies"]:
        if st.session_state.get('stop_run'):
            log.append("🛑 Run stopped by user during probing stage.")
            state["was_stopped"] = True
            break
        
        for i in range(3):
            if st.session_state.get('stop_run'): break
            
            log.append(f"- Running '{strategy['name']}' (Attempt {i+1}/3)...")
            st.session_state['comp_log'] = log
            
            result = run_single_jailbreak_attempt(
                task=state["task"],
                strategy=strategy,
                **state["model_config"],
                ui_placeholders=state.get("ui_monitor_placeholders")
            )
            results.append(result)
    
    if not state.get("was_stopped"): log.append("✅ Probing stage complete.")
    state["probing_results"] = results
    state["run_log"] = log
    st.session_state['comp_log'] = log
    return state

def analysis_stage_node(state: ComprehensiveRunState) -> ComprehensiveRunState:
    if state.get("was_stopped"): return state
    # ... (No changes to the logic inside this function)
    log = state["run_log"]
    log.append("\n**Stage 2: Analysis** - Identifying the best 4 strategies...")
    st.session_state['comp_log'] = log
    strategy_scores = {}
    for strategy in state["all_strategies"]:
        ratings = [r.get("final_rating", -1) for r in state["probing_results"] if r.get("strategy_id") == strategy.get("id") and r.get("final_rating", -1) != -1]
        if ratings: strategy_scores[strategy["id"]] = max(ratings)
        else: strategy_scores[strategy["id"]] = -1
    sorted_strategies = sorted(state["all_strategies"], key=lambda s: strategy_scores.get(s["id"], -1), reverse=True)
    top_4 = sorted_strategies[:4]
    state["top_4_strategies"] = top_4
    log.append("🏆 Top 4 Strategies identified:")
    for s in top_4: log.append(f"  - **{s['name']}** (Top score: {strategy_scores.get(s['id'], 'N/A')}/10)")
    log.append("✅ Analysis stage complete.")
    state["run_log"] = log
    st.session_state['comp_log'] = log
    return state

def combination_stage_node(state: ComprehensiveRunState) -> ComprehensiveRunState:
    if state.get("was_stopped") or len(state.get("top_4_strategies", [])) < 2: return state
    # ... (No changes to the logic inside this function)
    log = state["run_log"]
    log.append("\n**Stage 3: Combination** - Crafting two-pronged attacks...")
    st.session_state['comp_log'] = log
    combinations = list(itertools.combinations(state["top_4_strategies"], 2))
    state["strategy_combinations"] = combinations
    combined_prompts = []
    for strat_a, strat_b in combinations:
        if st.session_state.get('stop_run'):
            log.append("🛑 Run stopped by user during combination stage.")
            state["was_stopped"] = True
            break
        combo_name = f"{strat_a['name']} + {strat_b['name']}"
        log.append(f"- Combining '{combo_name}'...")
        st.session_state['comp_log'] = log
        prompt_for_combiner = f"..." # No change here
        raw_combo_response = query_ollama_model(model_name=state["model_config"]["crafter_model_name"], prompt=prompt_for_combiner, system_message=COMBINATION_CRAFTER_PROMPT)
        match = re.search(r'<prompt>(.*?)</prompt>', raw_combo_response, re.DOTALL)
        clean_prompt = match.group(1).strip() if match else raw_combo_response
        combined_prompts.append({"combo_name": combo_name, "prompt": clean_prompt})
    state["combined_prompts"] = combined_prompts
    if not state.get("was_stopped"): log.append("✅ Combination stage complete.")
    state["run_log"] = log
    st.session_state['comp_log'] = log
    return state

def assault_stage_node(state: ComprehensiveRunState) -> ComprehensiveRunState:
    if state.get("was_stopped") or not state.get("combined_prompts"): return state
    
    log = state["run_log"]
    log.append("\n**Stage 4: Final Assault** - Running combined attacks 3 times each...")
    st.session_state['comp_log'] = log
    
    results = []
    for combo in state["combined_prompts"]:
        if st.session_state.get('stop_run'):
            log.append("🛑 Run stopped by user during assault stage.")
            state["was_stopped"] = True
            break
            
        for i in range(3):
            if st.session_state.get('stop_run'): break
            
            log.append(f"- Running combined attack '{combo['combo_name']}' (Attempt {i+1}/3)...")
            st.session_state['comp_log'] = log
            
            # This is a "raw prompt" run, so we create a dummy strategy object
            dummy_strategy = {"id": combo["combo_name"], "name": combo["combo_name"]}
            
            # Here we need a way to run a raw prompt, not a strategy.
            # We'll adapt run_single_jailbreak_attempt's logic here directly.
            # This part is simplified for clarity, assuming a direct call to target and judge.
            from judge import evaluate_response_with_ai_judge # local import
            placeholders = state.get("ui_monitor_placeholders")
            if placeholders:
                placeholders["prompt_display"].markdown(f"**🤖 Combined Jailbreak Prompt:**\n```\n{combo['prompt']}\n```")
                placeholders["target_status"].info(f"🎯 Target model is processing...")
            
            target_response = query_ollama_model(state["model_config"]["target_model_name"], combo["prompt"])
            
            if placeholders:
                placeholders["response_display"].markdown(f"**🎯 Target LLM Response:**\n\n{target_response}")
                placeholders["judge_status"].info(f"🧑‍⚖️ Judge model is evaluating...")

            rating, reasoning = evaluate_response_with_ai_judge(state["task"]["prompt"], target_response, state["model_config"]["judge_model_name"])
            
            if placeholders:
                 # Logic for color from langgraph_setup.py
                verdict_color = "red" if rating >= 7 else "orange" if rating >= 4 else "green"
                placeholders["verdict_display"].markdown(f"**🧑‍⚖️ Judge's Rating:** :{verdict_color}[{rating}/10]\n\n**Reasoning:** {reasoning}")
            
            results.append({"strategy_name": combo['combo_name'], "final_rating": rating, "verdict_reasoning": reasoning})

    if not state.get("was_stopped"): log.append("✅ Final Assault stage complete.")
    state["final_assault_results"] = results
    state["run_log"] = log
    st.session_state['comp_log'] = log
    return state

# --- Graph Definition ---
def build_comprehensive_graph():
    # ... (No changes to the graph structure itself)
    workflow = StateGraph(ComprehensiveRunState)
    workflow.add_node("probing", probing_stage_node)
    workflow.add_node("analysis", analysis_stage_node)
    workflow.add_node("combination", combination_stage_node)
    workflow.add_node("assault", assault_stage_node)
    workflow.set_entry_point("probing")
    workflow.add_edge("probing", "analysis")
    workflow.add_edge("analysis", "combination")
    workflow.add_edge("combination", "assault")
    workflow.add_edge("assault", END)
    return workflow.compile()

comprehensive_graph = build_comprehensive_graph()