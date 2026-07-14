import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Config files
APP_DATA_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\baa0c5e4-4394-4ff3-be4e-a1a5ec181a62"
RAG_STORE_PATH = os.path.join(APP_DATA_DIR, "scratch", "rag_store.json")

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

def load_docs_from_files() -> list:
    """Dynamically loads and parses markdown files from the knowledge base directory."""
    docs = []
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logger.warning(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Returning empty list.")
        return []

    logger.info(f"Scanning knowledge base directory: {KNOWLEDGE_BASE_DIR}")
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Derive a pretty manual title (e.g. "📘 BP Store Operations Manual")
                manual_title = "BP Manual"
                first_line = content.split('\n')[0].strip()
                if first_line.startswith("# "):
                    manual_title = first_line.replace("# ", "").strip()
                
                # Split by section headings ("## ") to index independent sections
                sections = content.split("\n## ")
                header_context = sections[0].strip()
                
                for sec in sections[1:]:
                    lines = sec.split("\n")
                    sec_title = lines[0].strip()
                    sec_body = "\n".join(lines[1:]).strip()
                    
                    full_title = f"{manual_title} - {sec_title}"
                    category = manual_title.replace("📘 ", "").replace("📗 ", "").replace("📙 ", "").replace("📕 ", "").replace("📒 ", "").strip()
                    
                    docs.append({
                        "title": full_title,
                        "category": category,
                        "content": f"Source: {manual_title}\nSection: {sec_title}\n\nContext:\n{header_context}\n\nRules & Regulations:\n{sec_body}"
                    })
            except Exception as e:
                logger.error(f"Error reading RAG document '{filename}': {e}")
                
    logger.info(f"Dynamically loaded {len(docs)} sections from knowledge base directory.")
    return docs

RAG_DOCS = load_docs_from_files()

class RAGSystem:
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.db_docs = []
        self.embeddings_matrix = None
        self.load_or_build_index()

    def load_or_build_index(self):
        try:
            if os.path.exists(RAG_STORE_PATH):
                logger.info("Loading RAG store index from cache...")
                with open(RAG_STORE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if len(data.get("docs", [])) == len(RAG_DOCS):
                        self.db_docs = data["docs"]
                        self.embeddings_matrix = np.array(data["embeddings"])
                        return
                    else:
                        logger.info("RAG doc count mismatch. Rebuilding index...")
        except Exception as e:
            logger.warning(f"Failed to load cached RAG store: {e}. Rebuilding...")

        self.rebuild_index()

    def rebuild_index(self):
        logger.info("Rebuilding RAG index (calculating embeddings)...")
        os.makedirs(os.path.dirname(RAG_STORE_PATH), exist_ok=True)
        
        texts = [doc["content"] for doc in RAG_DOCS]
        try:
            logger.info("Generating embeddings using Google Generative AI...")
            vectors = self.embeddings.embed_documents(texts)
            self.db_docs = RAG_DOCS
            self.embeddings_matrix = np.array(vectors)
            
            # Save to cache
            cache_data = {
                "docs": self.db_docs,
                "embeddings": self.embeddings_matrix.tolist()
            }
            with open(RAG_STORE_PATH, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info("RAG index cached successfully.")
        except Exception as e:
            logger.error(f"Error creating RAG embeddings: {str(e)}", exc_info=True)

    def search(self, query: str, top_k: int = 2):
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return []
            
        try:
            query_vector = np.array(self.embeddings.embed_query(query))
            
            # Compute cosine similarities
            dot_product = np.dot(self.embeddings_matrix, query_vector)
            norms_matrix = np.linalg.norm(self.embeddings_matrix, axis=1)
            norm_query = np.linalg.norm(query_vector)
            
            similarities = dot_product / (norms_matrix * norm_query + 1e-9)
            
            # Get top K indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "doc": self.db_docs[idx],
                    "score": float(similarities[idx])
                })
            return results
        except Exception as e:
            logger.error(f"Error during RAG search: {str(e)}")
            return []

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    rag = RAGSystem()
    results = rag.search("what is the temp limit for milk in cafe?")
    for r in results:
        # Safely print on Windows terminal ignoring non-ASCII characters if not supported
        title_safe = r['doc']['title'].encode('ascii', errors='replace').decode('ascii')
        content_safe = r['doc']['content'][:200].encode('ascii', errors='replace').decode('ascii')
        print(f"Score: {r['score']:.4f} | Title: {title_safe}")
        print(content_safe + "...\n")
