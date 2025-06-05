# app.py
import streamlit as st
import pandas as pd
import uuid

from utils import load_tasks, load_strategies, add_strategy, load_results_log
from graph_runner import run_single_jailbreak_attempt, RESULTS_LOG_FILE

st.set_page_config(layout="wide", page_title="Make_A_Break")

st.title("🎭 Make_A_Break: Generative Jailbreak Workbench")
st.caption("Using a Crafter LLM, a Target LLM, and a Judge LLM via Ollama.")

# --- Session State Initialization ---
if 'tasks' not in st.session_state:
    st.session_state.tasks = load_tasks()
if 'strategies' not in st.session_state:
    st.session_state.strategies = load_strategies()
if 'results' not in st.session_state:
    st.session_state.results = load_results_log(RESULTS_LOG_FILE)
if 'running_test' not in st.session_state:
    st.session_state.running_test = False

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    crafter_model_name = st.text_input("Crafter Ollama Model", value="qwen3:8b")
    target_model_name = st.text_input("Target Ollama Model", value="qwen3:8b")
    judge_model_name = st.text_input("Judge Ollama Model", value="qwen3:8b")
    
    st.markdown("---")
    st.header("📝 Manage Jailbreak Strategies")
    
    with st.expander("Add New Generative Strategy"):
        new_strategy_name = st.text_input("Strategy Name")
        new_strategy_id = st.text_input("Strategy ID", value=f"custom_{uuid.uuid4().hex[:6]}")
        new_strategy_desc = st.text_area("Strategy Description (for display)")
        new_strategy_instruct = st.text_area("Instructions for Crafter LLM", height=150)
        
        if st.button("Add Strategy"):
            if all([new_strategy_name, new_strategy_id, new_strategy_desc, new_strategy_instruct]):
                try:
                    add_strategy({
                        "id": new_strategy_id, 
                        "name": new_strategy_name, 
                        "description": new_strategy_desc,
                        "instructions_for_crafter": new_strategy_instruct
                    })
                    st.session_state.strategies = load_strategies()
                    st.success(f"Strategy '{new_strategy_name}' added!")
                except ValueError as e:
                    st.error(str(e))
            else:
                st.warning("Please fill all fields for the new strategy.")

    st.subheader("Available Strategies")
    for strat in st.session_state.strategies:
        with st.expander(f"**{strat['name']}** (`{strat['id']}`)", expanded=False):
            st.markdown(f"**Description:** {strat.get('description', 'N/A')}")
            st.markdown(f"**Crafter Instructions:** {strat.get('instructions_for_crafter', 'N/A')}")

# --- Main Area ---
st.header("🚀 Run Jailbreak Tests")

col1, col2 = st.columns(2)
with col1:
    selected_task_ids = st.multiselect("Select Tasks:", [t['id'] for t in st.session_state.tasks], default=[st.session_state.tasks[0]['id']] if st.session_state.tasks else [])
with col2:
    selected_strategy_ids = st.multiselect("Select Strategies:", [s['id'] for s in st.session_state.strategies], default=[st.session_state.strategies[0]['id']] if st.session_state.strategies else [])

if st.button("Start Test Run", type="primary", use_container_width=True, disabled=st.session_state.running_test):
    if not selected_task_ids or not selected_strategy_ids:
        st.warning("Please select at least one task and one strategy.")
    else:
        st.session_state.running_test = True
        
        tasks_to_run = [t for t in st.session_state.tasks if t['id'] in selected_task_ids]
        strategies_to_apply = [s for s in st.session_state.strategies if s['id'] in selected_strategy_ids]
        
        total_runs = len(tasks_to_run) * len(strategies_to_apply)
        progress_bar = st.progress(0, text=f"Starting test run of {total_runs} combinations...")
        
        # --- Live Monitor Area ---
        st.subheader("🔴 Live Test Monitor")
        live_monitor_container = st.container(border=True)
        
        for i, task in enumerate(tasks_to_run):
            for j, strategy in enumerate(strategies_to_apply):
                current_run_num = (i * len(strategies_to_apply)) + j + 1
                progress_bar.progress(current_run_num / total_runs, text=f"Running {current_run_num}/{total_runs}: Task '{task['id']}' with Strategy '{strategy['name']}'")
                
                with live_monitor_container:
                    st.markdown(f"--- \n#### 🧪 Running Test {current_run_num}/{total_runs}: **{task['id']}** | **{strategy['name']}**")
                    
                    # Create placeholders for this specific run
                    ui_placeholders = {
                        "crafter_status": st.empty(),
                        "prompt_display": st.empty(),
                        "target_status": st.empty(),
                        "response_display": st.empty(),
                        "judge_status": st.empty(),
                        "verdict_display": st.empty()
                    }

                    try:
                        result = run_single_jailbreak_attempt(
                            task=task,
                            strategy=strategy,
                            target_model_name=target_model_name,
                            judge_model_name=judge_model_name,
                            crafter_model_name=crafter_model_name,
                            ui_placeholders=ui_placeholders
                        )
                        st.session_state.results.insert(0, result)

                    except Exception as e:
                        st.error(f"Critical framework error for {task['id']}/{strategy['id']}: {e}")
                        error_result = {"timestamp": pd.Timestamp.utcnow().isoformat(), "task_id": task['id'], "strategy_id": strategy['id'], "error_message": f"Framework error: {str(e)}", "final_verdict": "FRAMEWORK_ERROR"}
                        st.session_state.results.insert(0, error_result)
        
        progress_bar.progress(1.0, "Test run completed!")
        st.session_state.running_test = False
        st.balloons()
        st.rerun()

st.header("📜 Test Logs & Results")
if st.button("Refresh Results from Log File"):
    st.session_state.results = load_results_log(RESULTS_LOG_FILE)

if st.session_state.results:
    df_results = pd.DataFrame(st.session_state.results)
    st.dataframe(df_results[[
        "timestamp", "task_id", "strategy_name", "target_model_name", 
        "final_verdict", "verdict_reasoning", "error_message"
    ]], use_container_width=True)
    
    with st.expander("Show/Hide Full Results Log"):
        st.dataframe(df_results)

else:
    st.info("No results yet. Configure your models and run a test.")