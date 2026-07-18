"""Programmatic example demonstrating how to use the RAG system and Agent Graph directly in Python

Run this script:
    python example.py
"""

from langchain_core.messages import HumanMessage
from backend.agent.graph import agent_graph
from backend.rag.ingest import ingest_document
from backend.rag.retrieve import retrieve_context
from backend.memory.vectordb import vector_db

def main():
    print("--- 1. Programmatic RAG ingestion ---")
    # Clear vector db for demonstration purposes (optional)
    vector_db.clear()
    
    # Ingest text directly
    doc_text = "The quick brown fox jumps over the lazy dog. The project's top secret code is 007-GOLD."
    ingest_result = ingest_document(doc_text, source="sample_file.txt")
    print(f"Ingested chunks: {ingest_result['chunks_added']} (IDs: {ingest_result['chunk_ids']})")
    
    print("\n--- 2. Programmatic RAG retrieval ---")
    # Query vector store directly without going through the agent
    retrieved = retrieve_context("What is the secret code?", k=2)
    print("Retrieved snippets:")
    for snippet in retrieved:
        print(f" - {snippet}")

    print("\n--- 3. Programmatic Agent Graph execution ---")
    # Send a prompt to the agent. Because it mentions 'secret code',
    # the agent graph will automatically retrieve the relevant context and answer it.
    input_message = HumanMessage(content="Tell me what the secret code is according to the documents.")
    
    # Invoke the compiled LangGraph agent
    result = agent_graph.invoke({"messages": [input_message]})
    
    # Print agent response
    agent_reply = result["messages"][-1].content
    print(f"Agent Reply:\n{agent_reply}")

if __name__ == "__main__":
    main()
