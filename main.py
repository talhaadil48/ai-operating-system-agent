"""
Quick CLI to talk to the agent without spinning up the FastAPI server.
Good for testing the whole loop (agent -> router -> tool -> agent) fast.

Run:  python main.py
"""

from langchain_core.messages import HumanMessage

from backend.agent.graph import agent_graph
from backend.config import settings
from backend.logging_config import get_logger, log_timing, new_turn
from backend.memory.conversation import conversation_memory

log = get_logger(__name__)

SESSION_ID = "cli"


def main():
    print("AI OS — type 'exit' to quit.")
    print(
        f"[debug={'ON' if settings.DEBUG else 'off'}  "
        f"log_level={settings.LOG_LEVEL}  "
        f"(set DEBUG=true in .env, or run `DEBUG=true python main.py`, "
        f"for full prompt/tool/timing logs)]\n"
    )
    log.info("AI OS CLI started. debug=%s log_level=%s", settings.DEBUG, settings.LOG_LEVEL)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            log.info("CLI session ended by user.")
            break
        if not user_input:
            continue

        # Handle slash commands for document upload and stats in CLI
        if user_input.startswith(("/upload ", "/ingest ")):
            parts = user_input.split(" ", 1)
            filepath = parts[1].strip().strip('"').strip("'")
            import os
            if not os.path.exists(filepath):
                print(f"AI: (error) File not found: {filepath}\n")
                continue
            try:
                print(f"AI: Reading and ingesting '{filepath}'...")
                with open(filepath, "rb") as f:
                    file_data = f.read()
                
                from backend.rag.text import extract_text_from_bytes
                from backend.rag.ingest import ingest_document
                
                filename = os.path.basename(filepath)
                text = extract_text_from_bytes(file_data, filename)
                
                if not text.strip():
                    print("AI: (error) No text could be extracted from the file.\n")
                    continue
                
                result = ingest_document(text, source=filename)
                print(f"AI: Ingested '{filename}' successfully! Added {result['chunks_added']} chunk(s) to vector database.\n")
            except Exception as exc:
                log.exception("CLI failed to ingest document: %s", filepath)
                print(f"AI: (error) Failed to ingest file: {exc}\n")
            continue

        if user_input.lower() == "/stats":
            from backend.memory.vectordb import vector_db
            stats = vector_db.stats()
            print(f"AI: RAG Knowledge Base Stats:")
            print(f"  - Indexed Chunks: {stats['documents']}")
            print(f"  - Sources: {stats['sources']}")
            print(f"  - Database Location: {stats['store_path']}\n")
            continue

        if user_input.lower() == "/clear":
            from backend.memory.conversation import conversation_memory
            conversation_memory.clear(SESSION_ID)
            print("AI: Chat history cleared.\n")
            continue

        with new_turn(session_id=SESSION_ID) as turn_id:
            log.info("New turn %s. input=%r", turn_id, user_input)
            history = conversation_memory.get(SESSION_ID)

            try:
                with log_timing(log, "agent_graph.invoke", level=20):
                    result = agent_graph.invoke(
                        {"messages": history + [HumanMessage(content=user_input)]}
                    )
            except Exception as exc:
                log.exception("Agent graph invocation failed for turn %s", turn_id)
                print(f"AI: (error) something went wrong: {exc}\n")
                continue

            new_messages = result["messages"]

            conversation_memory.clear(SESSION_ID)
            conversation_memory.append(SESSION_ID, new_messages)

            print(f"AI: {new_messages[-1].content}\n")


if __name__ == "__main__":
    main()
