import sys
import time
import argparse
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import concurrent.futures

from agents.planner import plan_research
from agents.reader import read_papers
from agents.critic import critique
from agents.synthesizer import synthesize
from tools.arxiv_search import search_arxiv
from citations.bibtex import generate_bibtex
from src.pipeline import run_retrieval_benchmark

# Parse --no-eval flag if given
skip_eval = "--no-eval" in sys.argv

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📚 AI Research Assistant")
st.caption("Citation-grounded literature review with BibTeX export")

tab1, tab2 = st.tabs(["Research Pipeline", "Benchmark Mode"])

with tab1:
    topic = st.text_input(
        "Enter research topic",
        placeholder="Quantum Error Correction in Superconducting Qubits",
        key="pipeline_topic"
    )
    
    if st.button("Run Research Pipeline") and topic:
        with st.spinner("Planning research…"):
            plan = plan_research(topic)
            st.subheader("🧠 Research Plan")
            st.write(plan)
    
        with st.spinner("Searching papers…"):
            papers = search_arxiv(topic)
            st.subheader("📄 Papers Found")
            for p in papers:
                st.markdown(f"- **{p['title']}** ({p['year']})")
    
        with st.spinner("Reading & indexing papers…"):
            summaries = read_papers(papers)
    
        with st.spinner("Critiquing literature…"):
            crit = critique(topic)
            st.subheader("⚠️ Gaps & Limitations")
            st.write(crit)
    
        with st.spinner("Synthesizing report…"):
            report = synthesize(topic, papers)
            st.subheader("📝 Literature Review")
            st.write(report)
    
        bib_path = generate_bibtex(papers)
        with open(bib_path, "r", encoding="utf-8") as f:
            st.download_button(
                "⬇️ Download BibTeX",
                f.read(),
                file_name="references.bib"
            )

with tab2:
    st.header("Retrieval Strategy Benchmark")
    user_query = st.text_input("Enter a query to benchmark:", key="benchmark_query")
    ui_skip_eval = st.checkbox("Skip RAGAS evaluation (fast latency-only mode)", value=skip_eval)
    
    if st.button("Run Benchmark") and user_query:
        with st.spinner("Fetching corpus from arXiv for the query..."):
            papers = search_arxiv(user_query, max_results=20)
            docs = [p.get("summary", "") for p in papers]
            
        with st.spinner("Running 4 retrieval strategies in parallel..."):
            # run_retrieval_benchmark already runs strategies. We can modify it to run perfectly in parallel here:
            from src.retrieval.hybrid_retriever import HybridRetriever
            benchmark_retriever = HybridRetriever(strategy="hybrid")
            if docs:
                benchmark_retriever.add_docs(docs)
                
            strategies = ["dense", "sparse", "hybrid", "reranker"]
            
            def eval_strat(strat):
                benchmark_retriever.strategy = strat
                start = time.perf_counter()
                contexts = benchmark_retriever.retrieve(user_query, k=5)
                latency = time.perf_counter() - start
                
                from src.pipeline import evaluate_and_log
                dummy_answer = " ".join([c[:200] for c in contexts]) if contexts else "No contexts"
                
                scores = evaluate_and_log(
                    query=user_query, 
                    answer=dummy_answer, 
                    contexts=contexts, 
                    strategy=strat, 
                    skip_eval=ui_skip_eval,
                    latency=latency,
                    cost=200
                )
                return strat, scores, latency * 1000, contexts

            results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_strat = {executor.submit(eval_strat, s): s for s in strategies}
                for future in concurrent.futures.as_completed(future_to_strat):
                    strat, scores, lat, ctx = future.result()
                    results[strat] = {"scores": scores, "latency": lat, "contexts": ctx}
            
        st.success("Benchmark completed!")
        
        # Build Dataframe
        data = []
        best_strat_score = -1.0
        best_strat_name = None
        
        for strat in strategies:
            r = results[strat]
            scores = r["scores"]
            metric = (scores["faithfulness"] + scores["answer_relevancy"]) / 2
            if metric > best_strat_score:
                best_strat_score = metric
                best_strat_name = strat
                
            data.append({
                "Strategy": strat.capitalize(),
                "Faithfulness": scores["faithfulness"],
                "Answer Relevancy": scores["answer_relevancy"],
                "Context Precision": scores["context_precision"],
                "Context Recall": scores["context_recall"],
                "Latency (ms)": r["latency"],
                "Cost (tokens)": 200
            })
            
        df = pd.DataFrame(data)
        
        # Display Best Strategy
        st.markdown(f"### 🏆 Recommendation: **{best_strat_name.capitalize()}**")
        st.caption(f"Based on highest Faithfulness + Answer Relevancy ({best_strat_score:.2f})")
        
        # Display Table
        st.dataframe(df.style.highlight_max(axis=0, subset=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]).highlight_min(axis=0, subset=["Latency (ms)"]))
        
        col1, col2 = st.columns(2)
        with col1:
            # Latency Bar Chart
            fig_bar = px.bar(df, x="Strategy", y="Latency (ms)", title="Latency Comparison", color="Strategy")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            # Radar Chart
            fig_radar = go.Figure()
            for i, row in df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["Faithfulness"], row["Answer Relevancy"], row["Context Precision"], row["Context Recall"]],
                    theta=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                    fill='toself',
                    name=row["Strategy"]
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="RAGAS Metrics Radar"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.subheader("Retrieved Chunks Comparison")
        cols = st.columns(4)
        for i, strat in enumerate(strategies):
            with cols[i]:
                st.markdown(f"**{strat.capitalize()}**")
                ctxs = results[strat]["contexts"]
                for j, ctx in enumerate(ctxs):
                    with st.expander(f"Chunk {j+1}"):
                        st.write(ctx)
