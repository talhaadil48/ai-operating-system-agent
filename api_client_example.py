"""API Client Example demonstrating how to interact with the AI OS server via HTTP requests.

Make sure the server is running first:
    uvicorn backend.api.main:app --reload --port 8000
    
Then run this client example:
    python api_client_example.py
"""

import sys
import os

try:
    import requests
except ImportError:
    print("Error: 'requests' package is not installed. Run 'pip install requests' first.")
    sys.exit(1)

BASE_URL = "http://localhost:8000"

def main():
    print("Checking if API server is running...")
    try:
        health_check = requests.get(f"{BASE_URL}/health")
        health_check.raise_for_status()
        print("API Server Status:", health_check.json())
    except Exception as exc:
        print(f"Error: Could not connect to API server at {BASE_URL}. Is it running?")
        print("Start it with: uvicorn backend.api.main:app --reload")
        sys.exit(1)

    print("\n--- 1. Uploading text documents to RAG ---")
    text_data = {
        "source": "server_notes.txt",
        "text": "The secret code for the server API is API-999-SECRET."
    }
    response = requests.post(f"{BASE_URL}/rag/upload-text", json=text_data)
    print("Upload Text Response:", response.json())

    print("\n--- 2. Uploading a file to RAG (if exists) ---")
    # Let's write a dummy temp file to upload
    temp_filename = "temp_doc.txt"
    with open(temp_filename, "w") as f:
        f.write("Server deployment is scheduled for Sunday at 3:00 AM UTC.")
        
    try:
        with open(temp_filename, "rb") as f:
            files = {"file": (temp_filename, f, "text/plain")}
            data = {"source": "temp_doc.txt"}
            response = requests.post(f"{BASE_URL}/rag/upload-file", files=files, data=data)
            print("Upload File Response:", response.json())
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    print("\n--- 3. Querying the RAG database directly ---")
    search_data = {
        "query": "deployment schedule",
        "k": 3
    }
    response = requests.post(f"{BASE_URL}/rag/search", json=search_data)
    print("Search Results:")
    for result in response.json().get("results", []):
        print(" -", result)

    print("\n--- 4. Checking RAG statistics ---")
    response = requests.get(f"{BASE_URL}/rag/stats")
    print("Stats:", response.json())

    print("\n--- 5. Chatting with the Agent ---")
    chat_data = {
        "session_id": "test_session_123",
        "message": "What is the secret code for the server API?"
    }
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print("Agent Reply:", response.json().get("reply"))

if __name__ == "__main__":
    main()
