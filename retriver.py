from langchain_core.tools import tool
from langchain_core.documents import Document
from vector_database import vectorstore


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def search_pinecone(query: str) -> str:
    """Search for specific details in the uploaded documents. 
    Returns a string containing the text of the relevant documents."""
    
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        # If the retriever fails, we MUST return a string error message
        return f"Error retrieving documents: {str(e)}"
    
    if not docs:
        return "No relevant documents were found for this query."

    # Filter out empty strings before joining
    page_contents = [doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()]
    
    if not page_contents:
        return "Documents were found, but they contained no readable text content."

    content = "\n\n".join(page_contents)
    
    # Final safety check: OpenAI will 400 if this is empty
    return content if len(content) > 0 else "No text content available."
