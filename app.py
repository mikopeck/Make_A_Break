# app.py
import streamlit as st
import pandas as pd
import uuid # For generating unique IDs for new strategies

from utils import load_tasks, load_strategies, add_strategy, load_results_log
from graph_runner import run_single_jailbreak_attempt, RESULTS_LOG_FILE # Ensure graph_runner is importable

st.set_page_config(layout="wide", page_title="LLM Jailbreak Tester")

st.title("🤖 LLM Jailbreak Testing Workbench")
st.caption("Using local Ollama models, LangGraph, and an AI Judge.")

# --- Session State Initialization ---
if 'tasks' not in st.session_state:
    st.session_state.tasks = load_tasks()
if 'strategies' not in st.session_state:
    st.session_state.strategies = load_strategies()
if 'results' not in st.session_state:
    st.session_state.results = load_results_log(RESULTS_LOG_FILE)
if 'running_test' not in st.session_state:
    st.session_state.running_test = False


# --- Sidebar for Configuration and Strategy Management ---
with st.sidebar:
    st.header("⚙️ Configuration")
    target_model_name = st.text_input("Target Ollama Model Name", value="llama3:latest")
    judge_model_name = st.text_input("Judge Ollama Model Name", value="mistral:latest")
    
    st.markdown("---")
    st.header("📝 Manage Jailbreak Strategies")
    
    with st.expander("Add New Strategy"):
        new_strategy_name = st.text_input("Strategy Name (e.g., 'Polite Instruction')")
        new_strategy_id = st.text_input("Strategy ID (unique, e.g., 'S_polite_01')", value=f"custom_{uuid.uuid4().hex[:6]}")
        new_strategy_template = st.text_area("Strategy Template (use {task_prompt} for placeholder)", height=100)
        if st.button("Add Strategy"):
            if new_strategy_name and new_strategy_id and new_strategy_template:
                try:
                    add_strategy({"id": new_strategy_id, "name": new_strategy_name, "template": new_strategy_template})
                    st.session_state.strategies = load_strategies() # Refresh strategies
                    st.success(f"Strategy '{new_strategy_name}' added!")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to add strategy: {e}")
            else:
                st.warning("Please fill all fields for the new strategy.")

    st.subheader("Available Strategies")
    if st.session_state.strategies:
        for strat in st.session_state.strategies:
            st.markdown(f"- **{strat['name']}** (`{strat['id']}`)")
    else:
        st.caption("No strategies loaded. Add some or check `strategies.json`.")

# --- Main Area for Running Tests and Viewing Logs ---
st.header("🚀 Run Jailbreak Tests")

col1, col2 = st.columns(2)
with col1:
    selected_task_ids = st.multiselect(
        "Select Tasks to Run:",
        options=[task['id'] for task in st.session_state.tasks],
        default=[st.session_state.tasks[0]['id']] if st.session_state.tasks else []
    )
with col2:
    selected_strategy_ids = st.multiselect(
        "Select Strategies to Apply:",
        options=[strat['id'] for strat in st.session_state.strategies],
        default=[st.session_state.strategies[0]['id']] if st.session_state.strategies else []
    )

if st.button("Start Jailbreak Test Run", type="primary", disabled=st.session_state.running_test):
    if not selected_task_ids or not selected_strategy_ids:
        st.warning("Please select at least one task and one strategy.")
    else:
        st.session_state.running_test = True
        st.info(f"Starting test run for {len(selected_task_ids)} tasks and {len(selected_strategy_ids)} strategies...")
        
        tasks_to_run = [t for t in st.session_state.tasks if t['id'] in selected_task_ids]
        strategies_to_apply = [s for s in st.session_state.strategies if s['id'] in selected_strategy_ids]
        
        progress_bar = st.progress(0)
        total_runs = len(tasks_to_run) * len(strategies_to_apply)
        current_run = 0

        results_placeholder = st.empty() # For updating results dynamically

        for task in tasks_to_run:
            for strategy in strategies_to_apply:
                current_run += 1
                st.write(f"Running: Task '{task['id']}' with Strategy '{strategy['name']}'...")
                try:
                    result = run_single_jailbreak_attempt(
                        task=task,
                        strategy=strategy,
                        target_model_name=target_model_name,
                        judge_model_name=judge_model_name
                    )
                    # Prepend new result to the session state list for immediate display
                    st.session_state.results.insert(0, result)

                except Exception as e:
                    st.error(f"Critical error running test for {task['id']}/{strategy['id']}: {e}")
                    # Log a failed attempt if run_single_jailbreak_attempt itself fails
                    error_result = {
                        "timestamp": pd.Timestamp.utcnow().isoformat(), "task_id": task['id'], "strategy_id": strategy['id'],
                        "error_message": f"Framework error: {str(e)}", "final_verdict": "FRAMEWORK_ERROR"
                    }
                    st.session_state.results.insert(0, error_result)

                progress_bar.progress(current_run / total_runs)
                
                # Update displayed results (optional: convert to DataFrame for better display)
                if st.session_state.results:
                    df_results = pd.DataFrame(st.session_state.results)
                    results_placeholder.dataframe(df_results[[
                        "timestamp", "task_id", "strategy_name", "target_model_name", 
                        "final_verdict", "verdict_reasoning", "error_message"
                    ]], height=300)


        st.success("Test run completed!")
        st.session_state.running_test = False
        st.balloons()
        # Force a rerun to update the main display from file if needed, though session state should be fresh
        st.experimental_rerun() 


st.header("📜 Test Logs & Results")
if st.button("Refresh Results from Log File"):
    st.session_state.results = load_results_log(RESULTS_LOG_FILE)

if st.session_state.results:
    df_results = pd.DataFrame(st.session_state.results)
    
    # Display key columns
    st.dataframe(df_results[[
        "timestamp", "task_id", "strategy_name", "target_model_name", 
        "final_verdict", "verdict_reasoning", "error_message"
    ]], height=600)

    # Allow downloading results
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='jailbreak_test_results.csv',
        mime='text/csv',
    )
    
    st.subheader("Detailed View of Last Result")
    if st.session_state.results:
        st.json(st.session_state.results[0], expanded=False)

else:
    st.info("No results yet. Run some tests or check the log file.")