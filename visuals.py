# visuals.py
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

def update_visuals(placeholder: st.container, results_data: List[Dict[str, Any]]): # type: ignore
    """
    Clears a placeholder and redraws all result visualizations inside it.
    """
    if not results_data:
        return

    with placeholder.container():
        st.empty() # Clear previous visuals
        
        st.markdown("---")
        st.header("📊 Live Results Visualizations")

        df_results = pd.DataFrame(results_data)
        
        if 'final_rating' not in df_results.columns:
            df_results['final_rating'] = -1
        df_results['final_rating'] = pd.to_numeric(df_results['final_rating'], errors='coerce').fillna(-1).astype(int)

        # Filter out results with errors for meaningful visualizations
        df_viz = df_results[df_results['final_rating'] != -1].copy()

        if df_viz.empty:
            st.warning("No successful results yet to visualize...")
            return

        # --- The Visuals ---
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.subheader("Effectiveness per Strategy")
            if 'strategy_name' in df_viz.columns:
                strategy_effectiveness = df_viz.groupby('strategy_name')['final_rating'].mean().sort_values(ascending=False)
                st.bar_chart(strategy_effectiveness)
                st.caption("Average jailbreak rating (0-10) per strategy.")

        with viz_col2:
            st.subheader("Vulnerability per Task")
            if 'task_id' in df_viz.columns:
                task_vulnerability = df_viz.groupby('task_id')['final_rating'].mean().sort_values(ascending=False)
                st.bar_chart(task_vulnerability)
                st.caption("Average jailbreak rating (0-10) per task.")

        st.subheader("Heatmap: Strategy vs. Task Success")
        if 'strategy_name' in df_viz.columns and 'task_id' in df_viz.columns:
            try:
                heatmap_data = df_viz.pivot_table(
                    index='strategy_name', 
                    columns='task_id', 
                    values='final_rating',
                    aggfunc='mean'
                ).fillna(0)

                st.dataframe(
                    heatmap_data.style.background_gradient(cmap='viridis', axis=None).format("{:.1f}"),
                    use_container_width=True
                )
            except Exception:
                st.error("Could not generate heatmap.")