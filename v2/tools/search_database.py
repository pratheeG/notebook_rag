from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from v2.states.searchDBState import GradeRelevance
from v2.utils.llm import llm
from v2.utils.database import getVectorStore


grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", grader_system_prompt),
    ("human", "User Question: {query} \n\n Retrieved Content: {content}"),
])

# Use structured output to get a clean 'yes' or 'no'
structured_grader = llm.with_structured_output(GradeRelevance)
grader_chain = grader_prompt | structured_grader

@tool
def search_pinecone(query: str, config: RunnableConfig) -> str:
    """Search for specific details in the uploaded documents."""
    print(f"search_pinecone called with query: {query}")
    thread_id = config["configurable"].get("thread_id", "")
    vectorStore = getVectorStore(namespace=thread_id)
    retriever = vectorStore.as_retriever(search_kwargs={"k": 2})

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        docs = []
    
    if not docs:
        return "No documents matched"

    # Extract and join content
    page_contents = [doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()]
    content = "\n\n".join(page_contents)
    
    if not content.strip():
        return "No documents matched"

    # --- SEMANTIC CHECK ---
    try:
        grading_result = grader_chain.invoke({"query": query, "content": content})
        
        if grading_result.binary_score.lower() == "yes":
            return content
        else:
            return "No documents matched"
            
    except Exception as e:
        print(f"Grader failed: {e}. Defaulting to returning content.")
        return content