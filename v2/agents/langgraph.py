

from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from v2.utils.memory import memory
from v2.utils.llm import llm
from v2.agents.nodes import ask_user_permission, grade_results
from v2.agents.state import getState
from v2.states.chatbotState import ChatBotState
from v2.tools.search_database import search_pinecone
from v2.tools.summarizer import summarize_document_tool


tools = [search_pinecone, summarize_document_tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: ChatBotState) -> ChatBotState:
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
    
    for msg in full_messages:
        if type(msg).__name__ == "ToolMessage" and not msg.content:
            msg.content = "No data returned from tool."

    return {"messages": [llm_with_tools.invoke(full_messages)]}

builder = getState(ChatBotState)

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