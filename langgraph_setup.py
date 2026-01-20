
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List

from llm import llm
from memory import memory
from retriver import search_pinecone

tools = [search_pinecone]

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> State:
    messages = state["messages"]
    
    # Fix: Ensure no ToolMessages have empty content before sending to LLM
    for msg in messages:
        print(msg.content)
        if type(msg).__name__ == "ToolMessage" and not msg.content:
            msg.content = "No data returned from tool."

    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot_node")
builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")

graph = builder.compile(checkpointer=memory)