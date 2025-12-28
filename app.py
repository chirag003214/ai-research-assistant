import streamlit as st
from agents.planner import plan_research
from agents.reader import read_papers
from agents.critic import critique
from agents.synthesizer import synthesize
from tools.arxiv_search import search_arxiv
from citations.bibtex import generate_bibtex

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📚 AI Research Assistant (Physics & ML)")
st.caption("Citation-grounded literature review with BibTeX export")

topic = st.text_input(
    "Enter research topic",
    placeholder="Quantum Error Correction in Superconducting Qubits"
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
    with open(bib_path, "r") as f:
        st.download_button(
            "⬇️ Download BibTeX",
            f.read(),
            file_name="references.bib"
        )
