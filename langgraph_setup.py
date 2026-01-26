
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List
from langchain_core.messages import SystemMessage

from llm import llm
from memory import memory
from retriver import search_pinecone
from summary import summarize_document_tool

tools = [search_pinecone, summarize_document_tool]

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_permission: bool # Track if we are waiting for user confirmation

def grade_results(state: State):
    last_message = state["messages"][-1]
    
    tool_called = getattr(last_message, "name", "")

    # 2. If it's a summary, we don't grade it. Just go back to the chatbot to present it.
    if tool_called == "summarize_document_tool" and len(last_message.content) > 5:
        return "generate_answer"
    
    # After 'tools' node, the last message is a ToolMessage
    # We check if the search failed or returned the 'No documents' string
    if "No documents matched" in last_message.content or len(last_message.content) < 5:
        return "ask_user"
    
    # If successful, go back to chatbot so it can summarize the found info
    return "generate_answer"

def ask_user_permission(state: State) -> State:
    # We update the state to indicate we are now in 'permission mode'
    return {
        "messages": [("assistant", "I couldn't find specific details in your documents. Would you like me to answer using my general AI knowledge instead?")],
        "ask_permission": True 
    }

def chatbot(state: State) -> State:
    messages = state["messages"]
    system_message = SystemMessage(content=
        "You are a helpful assistant that answers questions based ONLY on uploaded documents. "
        "You have two specific tools at your disposal:\n"
            "1. 'search_pinecone': Use this for specific factual questions (e.g., 'What is the price of X?').\n"
            "2. 'summarize_document_tool': Use this for general requests (e.g., 'Summarize this', 'Give me an overview').\n\n"
        "Do not call both at once. Choose the one that fits the user's intent."
        "If the tool returns str having the 'No documents matched' or 'No documents found to summarize', do not make up an answer. "
    )

    full_messages = [system_message] + messages
    
    # Fix: Ensure no ToolMessages have empty content before sending to LLM
    for msg in full_messages:
        if type(msg).__name__ == "ToolMessage" and not msg.content:
            msg.content = "No data returned from tool."

    return {"messages": [llm_with_tools.invoke(full_messages)]}

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_node("ask_permission", ask_user_permission)

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_conditional_edges(
    "tools",
    grade_results,
    {
        "ask_user": "ask_permission",
        "generate_answer": "chatbot_node"
    }
)
builder.add_edge("ask_permission", END)

graph = builder.compile(checkpointer=memory)