from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ChatBotState(TypedDict):
    messages: Annotated[list, add_messages]
    ask_permission: bool # Track if we are waiting for user confirmation