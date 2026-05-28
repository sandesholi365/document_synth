import streamlit as st
from agents import build_graph, AgentState

st.set_page_config(page_title="Research Synthesizer", layout="wide")
st.title("Multi-Agent Research Paper Synthesizer")
st.markdown("Enter a research topic. Four AI agents will search, summarise, critique, and synthesise a report for you.")

topic = st.text_input("Research topic", placeholder="e.g., autonomous agents in reinforcement learning")

if st.button("Generate Report") and topic:
    with st.spinner("Agents are working... This may take a minute."):
        app = build_graph()
        initial_state = AgentState(
            topic=topic,
            papers=[],
            summary="",
            critique="",
            final_report="",
            messages=[]
        )
        result = app.invoke(initial_state)
        
        st.subheader("Searched Papers")
        for i, paper in enumerate(result['papers'], 1):
            st.write(f"**{i}. {paper['title']}**")
            st.write(f"*{paper['url']}*")
        
        st.subheader("Initial Summary")
        st.text_area("Summary", result['summary'], height=200)
        
        st.subheader("Peer Critique")
        st.text_area("Critique", result['critique'], height=200)
        
        st.subheader("Final Synthesised Report")
        st.markdown(result['final_report'])