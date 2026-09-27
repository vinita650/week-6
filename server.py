from pathlib import Path
import tomllib
import chromadb
from openai import OpenAI
from mcp.server import MCPServer

CHROMA_DIR = Path.home() / "Documents" / "week-4" / "chroma_db"
SECRETS = Path.home() / "Documents" / "week-4" / ".streamlit" / "secrets.toml"

with open(SECRETS, "rb") as f:
    secrets = tomllib.load(f)
    openai_client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

mcp = MCPServer("my-docs")

@mcp.tool()
def ping() -> str:
    """Connection test that returns a short confirmation message."""
    return "my-docs server is alive"

@mcp.tool()
def index_stats() -> str:
    """Reports how many chunks are stored in Vinita's card rulebook index. Used to check the document index is loaded."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name="docs")
        count = collection.count()
        return f"The card rulebook index contains {count} chunks."
    except Exception as e:
        return str(e)

@mcp.tool()
def search_my_docs(question: str) -> str:
    """Search Vinita's RBI credit and debit card rulebook and return the matching passages with their page numbers. Use this for ANY question about credit or debit card rules, limits, fees, or charges."""
    try:
        response = openai_client.embeddings.create(model="text-embedding-3-small", input=question)
        vector = response.data[0].embedding

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_collection(name="docs")

        results = collection.query(query_embeddings=[vector], n_results=3)

        output = []
        for i, (doc, metadata, distance) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
            page_num = metadata.get("page", "unknown")
            dist_rounded = round(distance, 2)
            passage = doc[:400]
            output.append(f"[Page {page_num}, distance: {dist_rounded}]\n{passage}")

        return "\n\n".join(output)
    except Exception as e:
        return str(e)

@mcp.tool()
def list_my_topics() -> str:
    """Extracts and returns the main subjects covered in the RBI credit and debit card rulebook. Use this to explore what topics are available in the document, like fees, limits, fraud protection, or dispute resolution."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name="docs")

        all_docs = collection.get()
        sample = all_docs["documents"][:15]
        combined_text = " ".join(sample)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the 8-10 main topics or subjects from this text about RBI credit and debit card rules. Return only a comma-separated list of topics:\n\n{combined_text}"
                }
            ],
            temperature=0
        )

        topics = response.choices[0].message.content.strip()
        return f"Main topics in the RBI rulebook: {topics}"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    mcp.run(transport="stdio")
