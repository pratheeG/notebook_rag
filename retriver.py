from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import llm # Ensure your LLM is imported
from vector_database import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 1. Define the output structure for the grader
class GradeRelevance(BaseModel):
    binary_score: str = Field(
        description="Is the retrieved content relevant to the query? 'yes' or 'no'"
    )

# 2. Create the Grader chain
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
def search_pinecone(query: str) -> str:
    """Search for specific details in the uploaded documents."""
    
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"
    
    if not docs:
        return "No documents matched"

    # Extract and join content
    page_contents = [doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()]
    content = "\n\n".join(page_contents)
    
    if not content.strip():
        return "No documents matched"

    # --- SEMANTIC CHECK ---
    print(f"--- SEMANTIC CHECK FOR QUERY: {query} ---")
    try:
        grading_result = grader_chain.invoke({"query": query, "content": content})
        
        if grading_result.binary_score.lower() == "yes":
            print("--- GRADE: RELEVANT ---")
            return content
        else:
            print("--- GRADE: NOT RELEVANT ---")
            return "No documents matched"
            
    except Exception as e:
        print(f"Grader failed: {e}. Defaulting to returning content.")
        return content