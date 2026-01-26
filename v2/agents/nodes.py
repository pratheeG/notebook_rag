from v2.states.chatbotState import ChatBotState
from v2.states.summaryState import SummaryState
from v2.utils.llm import llm


def map_summaries(state: SummaryState):
    prompt = "Summarize the following document chunk concisely:\n\n{content}"
    new_summaries = []
    
    for content in state["contents"]:
        response = llm.invoke(prompt.format(content=content))
        new_summaries.append(response.content)
    
    return {"summaries": new_summaries}

def reduce_summaries(state: SummaryState):
    combined_text = "\n\n".join(state["summaries"])
    prompt = f"Combine these summaries into one cohesive final report:\n\n{combined_text}"
    
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

def grade_results(state: ChatBotState):
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

def ask_user_permission(state: ChatBotState) -> ChatBotState:
    # We update the state to indicate we are now in 'permission mode'
    return {
        "messages": [("assistant", "I couldn't find specific details in your documents. Would you like me to answer using my general AI knowledge instead?")],
        "ask_permission": True 
    }