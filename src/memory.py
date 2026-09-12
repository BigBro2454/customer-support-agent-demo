import os
import glob
import chromadb
try:
    from src.config import CONFIG
except ModuleNotFoundError:
    from config import CONFIG

class MemoryModule:
    """
    Memory Module handles the semantic memory of the agent using a Vector Store (ChromaDB).
    It stores and retrieves knowledge base articles based on relevance.
    """
    def __init__(self, persist_dir: str | None = None):
        # Anchor persistent ChromaDB client to project root or specified directory
        if persist_dir:
            chroma_path = persist_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            chroma_path = os.path.join(base_dir, "chroma_db")

        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Recreate the collection to ensure it's fresh (for learning purposes)
        try:
            self.client.delete_collection(name=CONFIG["CHROMA_COLLECTION"])
        except Exception:
            pass # Collection doesn't exist yet
            
        self.collection = self.client.create_collection(name=CONFIG["CHROMA_COLLECTION"])
        
        print("💾 [Memory] Initializing Vector Store with Knowledge Base...")
        self._load_knowledge_base()
        
    def _load_knowledge_base(self):
        """
        Loads all markdown files from the knowledge base directory into ChromaDB.
        """
        kb_path = os.path.join(os.path.dirname(__file__), "..", CONFIG["KB_DIR"])
        
        # Find all .md files in the kb directory
        md_files = glob.glob(os.path.join(kb_path, "*.md"))
        
        if not md_files:
            print(f"⚠️ [Memory] No knowledge base files found in {kb_path}!")
            return
            
        documents = []
        metadatas = []
        ids = []
        
        for i, file_path in enumerate(md_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            filename = os.path.basename(file_path)
            
            # For simplicity, we are chunking by whole document. 
            # In a real system, you might chunk by paragraph or section.
            documents.append(content)
            metadatas.append({"source": filename})
            ids.append(f"doc_{i}")
            
        # Add documents to ChromaDB. ChromaDB will automatically handle embedding them
        # using its default embedding model (all-MiniLM-L6-v2).
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"💾 [Memory] Added {len(documents)} documents to the Vector Store.")
        
    def retrieve(self, query: str, n_results: int = 2) -> list[str]:
        """
        Retrieves the top-k most relevant documents from the vector store based on the query.
        """
        print(f"💾 [Memory] Retrieving relevant policies for: '{query}'...")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract the document strings from the results
        documents = results["documents"][0] if results["documents"] else []
        return documents
