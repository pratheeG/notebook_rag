
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List

from llm import llm
from memory import memory


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> State:
    print("Chatbot node invoked")
    print("ll ", llm)
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)

builder.add_edge(START, "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile(checkpointer=memory)