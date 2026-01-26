from typing import List, TypedDict
from llm import llm
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import tool

from vector_database import vectorstore


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

class SummaryState(TypedDict):
    contents: List[str]      # The raw document chunks
    summaries: List[str]     # The individual "mapped" summaries
    final_summary: str       # The final consolidated result

# Compile the sub-graph

# Node 1: Map - Generate individual summaries
def map_summaries(state: SummaryState):
    prompt = "Summarize the following document chunk concisely:\n\n{content}"
    new_summaries = []
    
    for content in state["contents"]:
        response = llm.invoke(prompt.format(content=content))
        new_summaries.append(response.content)
    
    return {"summaries": new_summaries}

# Node 2: Reduce - Consolidate into one
def reduce_summaries(state: SummaryState):
    combined_text = "\n\n".join(state["summaries"])
    prompt = f"Combine these summaries into one cohesive final report:\n\n{combined_text}"
    
    response = llm.invoke(prompt)
    return {"final_summary": response.content}


# ... (existing imports and search_pinecone code) ...

# 1. Define the Sub-Graph for Summarization
summary_workflow = StateGraph(SummaryState)
summary_workflow.add_node("map_node", map_summaries)
summary_workflow.add_node("reduce_node", reduce_summaries)
summary_workflow.add_edge(START, "map_node")
summary_workflow.add_edge("map_node", "reduce_node")
summary_workflow.add_edge("reduce_node", END)

summary_app = summary_workflow.compile()

@tool
def summarize_document_tool(query: str) -> str:
    """Use this when the user asks for a general summary, a TL;DR, or an overview of the documents.
    Pass a descriptive query like 'main themes' if the user just says 'summarize'."""
    
    print(f"Summarizing documents for query: {query}")
    # 1. Fetch more chunks for a better summary (k=10 or 15)
    # We use a blank query or the user's query to get broad content
    docs = retriever.vectorstore.as_retriever(search_kwargs={"k": 100}).invoke("")
    contents = [d.page_content for d in docs if d.page_content]

    print(f"Fetched {len(contents)} document chunks for summarization.")
    
    if not contents:
        return "No documents found to summarize."
    
    # 2. Run the compiled sub-graph
    initial_state = {"contents": contents, "summaries": [], "final_summary": ""}
    result = summary_app.invoke(initial_state)
    
    print("Summarization complete.")
    print(f"Final Summary: {result['final_summary']}")
    return result["final_summary"]