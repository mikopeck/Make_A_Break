# app.py
import streamlit as st
import pandas as pd
import uuid

from utils import load_tasks, load_strategies, add_strategy, load_results_log
from graph_runner import run_single_jailbreak_attempt
# Import the new comprehensive runner
from comprehensive_runner import comprehensive_graph, ComprehensiveRunState

st.set_page_config(layout="wide", page_title="Make_A_Break")

st.title("🎭 Make_A_Break: Generative Jailbreak Workbench")
st.caption("Using a Crafter LLM, a Target LLM, and a Judge LLM via Ollama.")

# --- Session State & Callbacks ---
if 'tasks' not in st.session_state: st.session_state.tasks = load_tasks()
if 'strategies' not in st.session_state: st.session_state.strategies = load_strategies()
if 'results' not in st.session_state: st.session_state.results = load_results_log("results/jailbreak_log.jsonl")
if 'running_test' not in st.session_state: st.session_state.running_test = False
if 'stop_run' not in st.session_state: st.session_state.stop_run = False
if 'comp_log' not in st.session_state: st.session_state.comp_log = []

def stop_button_callback():
    """Sets the flag to stop the current run."""
    st.session_state.stop_run = True

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
                    add_strategy({"id": new_strategy_id, "name": new_strategy_name, "description": new_strategy_desc, "instructions_for_crafter": new_strategy_instruct})
                    st.session_state.strategies = load_strategies()
                    st.success(f"Strategy '{new_strategy_name}' added!")
                except ValueError as e: st.error(str(e))
            else: st.warning("Please fill all fields for the new strategy.")

    st.subheader("Available Strategies")
    strategies_list = st.session_state.get('strategies', [])
    for strat in strategies_list:
        with st.expander(f"**{strat['name']}** (`{strat['id']}`)", expanded=False):
            st.markdown(f"**Description:** {strat.get('description', 'N/A')}")
            st.markdown(f"**Crafter Instructions:** {strat.get('instructions_for_crafter', 'N/A')}")

# --- Main Area ---
st.header("🚀 Run Tests")

# --- Stop Button ---
if st.session_state.running_test:
    st.button("🛑 Stop Run", type="primary", on_click=stop_button_callback, use_container_width=True, help="Stops the current test run after the current step finishes.")

comp_mode = st.checkbox("Enable Comprehensive Run Mode", help="Runs a multi-stage attack: Probes all strategies, finds the top 4, combines them into new attacks, and runs them.", disabled=st.session_state.running_test)
st.markdown("---")

if comp_mode:
    st.subheader("🔬 Comprehensive Run Configuration")
    st.info("This mode will run on a single task, testing all available strategies to find the best combinations.")
    selected_task_id = st.selectbox("Select a Single Task for the Comprehensive Run:", [t['id'] for t in st.session_state.get('tasks', [])], disabled=st.session_state.running_test)
    
    if st.button("Start Comprehensive Run", type="secondary", use_container_width=True, disabled=st.session_state.running_test):
        st.session_state.running_test = True
        st.session_state.stop_run = False
        st.session_state.comp_log = ["Initializing Comprehensive Run..."]

        task_to_run = next((t for t in st.session_state.tasks if t['id'] == selected_task_id), None)
        model_config = {"target_model_name": target_model_name, "judge_model_name": judge_model_name, "crafter_model_name": crafter_model_name}
        
        st.subheader("🔴 Live Monitor")
        live_monitor_container = st.container(border=True)
        with live_monitor_container:
            ui_placeholders = {
                "crafter_status": st.empty(), "prompt_display": st.empty(), "target_status": st.empty(), 
                "response_display": st.empty(), "judge_status": st.empty(), "verdict_display": st.empty()
            }
        
        initial_state = ComprehensiveRunState(
            task=task_to_run, all_strategies=st.session_state.get('strategies', []), model_config=model_config,
            ui_monitor_placeholders=ui_placeholders, was_stopped=False, run_log=[],
            probing_results=[], top_4_strategies=[], strategy_combinations=[], combined_prompts=[], final_assault_results=[]
        )
        
        with st.spinner("Comprehensive run in progress... This may take a very long time."):
            final_state = comprehensive_graph.invoke(initial_state)

        if final_state.get("was_stopped"):
            st.warning("Run was stopped by the user.")
        else:
            st.success("Comprehensive Run Finished!")
        
        st.session_state.running_test = False
        st.rerun()

    st.subheader("Comprehensive Run Log")
    log_container = st.container(height=400)
    log_container.markdown("  \n".join(st.session_state.comp_log))

else: # Standard Mode
    st.subheader("🎯 Standard Run Configuration")
    col1, col2 = st.columns(2)
    with col1: selected_task_ids = st.multiselect("Select Tasks:", [t['id'] for t in st.session_state.get('tasks', [])], default=[st.session_state.tasks[0]['id']] if st.session_state.tasks else [], disabled=st.session_state.running_test)
    with col2: selected_strategy_ids = st.multiselect("Select Strategies:", [s['id'] for s in st.session_state.get('strategies', [])], default=[s['id'] for s in st.session_state.strategies], disabled=st.session_state.running_test)

    if st.button("Start Standard Test Run", type="secondary", use_container_width=True, disabled=st.session_state.running_test):
        if not selected_task_ids or not selected_strategy_ids:
            st.warning("Please select at least one task and one strategy.")
        else:
            st.session_state.running_test = True
            st.session_state.stop_run = False
            tasks_to_run = [t for t in st.session_state.tasks if t['id'] in selected_task_ids]
            strategies_to_apply = [s for s in st.session_state.strategies if s['id'] in selected_strategy_ids]
            total_runs = len(tasks_to_run) * len(strategies_to_apply)
            progress_bar = st.progress(0, text=f"Starting test run of {total_runs} combinations...")
            
            st.subheader("🔴 Live Test Monitor")
            live_monitor_container = st.container(border=True)
            
            for i, task in enumerate(tasks_to_run):
                if st.session_state.stop_run: break
                for j, strategy in enumerate(strategies_to_apply):
                    if st.session_state.stop_run: break
                    current_run_num = (i * len(strategies_to_apply)) + j + 1
                    progress_bar.progress(current_run_num / total_runs, text=f"Running {current_run_num}/{total_runs}: Task '{task['id']}' with Strategy '{strategy['name']}'")
                    with live_monitor_container:
                        st.markdown(f"--- \n#### 🧪 Running Test {current_run_num}/{total_runs}: **{task['id']}** | **{strategy['name']}**")
                        ui_placeholders = {"crafter_status": st.empty(), "prompt_display": st.empty(), "target_status": st.empty(), "response_display": st.empty(), "judge_status": st.empty(), "verdict_display": st.empty()}
                        result = run_single_jailbreak_attempt(task=task, strategy=strategy, target_model_name=target_model_name, judge_model_name=judge_model_name, crafter_model_name=crafter_model_name, ui_placeholders=ui_placeholders)
                        st.session_state.results.insert(0, result)

            if st.session_state.stop_run:
                st.warning("Run stopped by user.")
            else:
                progress_bar.progress(1.0, "Test run completed!")
                st.balloons()
            
            st.session_state.running_test = False
            st.rerun()

st.markdown("---")
st.header("📜 Test Logs & Results (Standard Runs)")
if st.button("Refresh Results from Log File"):
    st.session_state.results = load_results_log("results/jailbreak_log.jsonl")

if st.session_state.results:
    df_results = pd.DataFrame(st.session_state.results)
    if 'final_rating' not in df_results.columns:
        df_results['final_rating'] = -1
    df_results['final_rating'] = pd.to_numeric(df_results['final_rating'], errors='coerce').fillna(-1).astype(int)

    st.dataframe(df_results[[
        "timestamp", "task_id", "strategy_name", "final_rating", "verdict_reasoning", "error_message"
    ]], use_container_width=True)
    
    with st.expander("Show/Hide Full Results Log"):
        st.dataframe(df_results)
else:
    st.info("No standard run results yet.")