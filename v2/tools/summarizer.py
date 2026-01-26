from v2.agents.nodes import map_summaries, reduce_summaries
from v2.agents.state import getState
from v2.states.summaryState import SummaryState
from langgraph.graph import  END, START
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from v2.utils.database import getVectorStore

summary_workflow = getState(SummaryState)
summary_workflow.add_node("map_node", map_summaries)
summary_workflow.add_node("reduce_node", reduce_summaries)

summary_workflow.add_edge(START, "map_node")
summary_workflow.add_edge("map_node", "reduce_node")
summary_workflow.add_edge("reduce_node", END)

summary_app = summary_workflow.compile()

@tool
def summarize_document_tool(query: str, config: RunnableConfig) -> str:
    """Use this when the user asks for a general summary, a TL;DR, or an overview of the documents.
    Pass a descriptive query like 'main themes' if the user just says 'summarize'."""

    thread_id = config["configurable"].get("thread_id", "")
    vectorStore = getVectorStore(namespace=thread_id)
    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})
    
    docs = retriever.vectorstore.as_retriever(search_kwargs={"k": 100}).invoke("")
    contents = [d.page_content for d in docs if d.page_content]

    if not contents:
        return "No documents found to summarize."
    
    initial_state = {"contents": contents, "summaries": [], "final_summary": ""}
    result = summary_app.invoke(initial_state)
    
    return result["final_summary"]