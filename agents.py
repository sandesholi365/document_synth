import os
import json
from typing import List, Dict, Any, Annotated
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
import arxiv
from typing import TypedDict

class AgentState(TypedDict):
    topic: str
    papers: List[Dict[str, str]]
    summary: str
    critique: str
    final_report: str
    messages: Annotated[List, add_messages]

@tool
def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search arXiv for the given query and return a list of paper info."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "summary": paper.summary.replace('\n', ' '),
            "url": paper.entry_id
        })
    return results

def get_llm():
    if os.getenv("GROQ_API_KEY"):
        return ChatGroq(model="llama3-70b-8192", temperature=0.2, groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        return ChatOllama(model="llama3.2", temperature=0.2)

llm = get_llm()

def searcher_node(state: AgentState) -> AgentState:
    topic = state['topic']
    papers = search_arxiv.invoke(topic)
    state['papers'] = papers
    papers_text = "\n\n".join([f"Title: {p['title']}\nAbstract: {p['summary']}\nURL: {p['url']}" for p in papers])
    state['messages'] = [SystemMessage(content="You are a research assistant. Here are the papers found:"),
                         HumanMessage(content=papers_text)]
    return state

def summarizer_node(state: AgentState) -> AgentState:
    prompt = ("You are an expert research summarizer. Based on the paper abstracts above, "
              "write a concise bullet-point summary covering the main findings, methodologies, and contributions. "
              "Keep it under 300 words.")
    messages = state['messages'] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    state['summary'] = response.content
    state['messages'] = messages + [response]
    return state

def critic_node(state: AgentState) -> AgentState:
    prompt = ("You are a rigorous peer reviewer. Read the summary below and identify any overclaims, "
              "missing context, or areas that need deeper investigation. Be constructive but sharp."
              f"\n\nSUMMARY:\n{state['summary']}")
    messages = [SystemMessage(content="You are a critical reviewer."), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    state['critique'] = response.content
    state['messages'] = messages + [response]
    return state

def synthesizer_node(state: AgentState) -> AgentState:
    prompt = ("You are a senior research lead. Given the original paper information, the summary, and the critique, "
              "produce a final research digest that is balanced, accurate, and ready to be read by a student. "
              "Structure it with an introduction, key insights, and a conclusion. "
              f"\n\nPAPERS:\n{json.dumps(state['papers'], indent=2)}"
              f"\n\nSUMMARY:\n{state['summary']}"
              f"\n\nCRITIQUE:\n{state['critique']}")
    messages = [SystemMessage(content="You are a senior research lead."), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    state['final_report'] = response.content
    return state

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.set_entry_point("searcher")
    workflow.add_edge("searcher", "summarizer")
    workflow.add_edge("summarizer", "critic")
    workflow.add_edge("critic", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()