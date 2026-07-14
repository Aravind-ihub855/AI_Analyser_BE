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

RAG_DOCS = [
    {
        "title": "Cold Chain & Refrigeration Temperature Policy",
        "category": "Food Safety",
        "content": """Standard Operating Procedure (SOP) for Wild Bean Cafe Cold Chain:
        All refrigeration units storing milk, fresh sandwiches, and other dairy items must maintain temperatures strictly between 1.0°C and 4.0°C.
        If a refrigeration unit temp sensor reports between 4.1°C and 5.0°C, a 'Warning' status is triggered, requiring a visual inspection within 30 minutes.
        If a refrigeration unit temp exceeds 5.0°C for more than 30 consecutive minutes, a 'Critical' alarm is activated. All fresh products must be moved to an alternative freezer or discarded immediately if left for more than 1 hour, and a service technician must be dispatched immediately.
        Records of temperature inspections must be kept for 3 months for audit compliance."""
    },
    {
        "title": "Fuel Spill Emergency Response Plan",
        "category": "Safety & Environment",
        "content": """Emergency Response Plan for Fuel Spills on Forecourt:
        Minor Spills (under 5 Litres):
        1. Immediately stop the fuel dispenser using the safety switch.
        2. Apply spill kit absorbent powder/kitty litter over the spill.
        3. Sweep up the absorbent and place it in the designated hazardous waste bin. Do not wash fuel into storm drains.
        
        Major Spills (over 5 Litres):
        1. Immediately press the Emergency Stop Button (E-Stop) to isolate all dispensers.
        2. Evacuate the forecourt and cordon off the area.
        3. Dispatch fire extinguishers if a fire hazard is present.
        4. Contact the emergency services (911 / Fire department) and the BP incident response hotline at 1800-555-SAFE.
        5. Log the incident in the compliance logs within 2 hours."""
    },
    {
        "title": "Store Procurement & Auto-Reordering Guidelines",
        "category": "Inventory Management",
        "content": """BP Store Automatic Reordering Policy:
        Inventory is monitored daily. Reorder point (ROL) triggers automatic purchase order generation.
        The Reorder Point (ROL) is calculated as: ROL = (Average Daily Consumption * Lead Time in Days) + Safety Stock Level.
        When the stock level falls below ROL, a PO is triggered automatically with a Reorder Quantity (ROQ) calculated as: ROQ = Capacity - Stock Level.
        Pending orders must be reviewed by the Store Manager before final sign-off.
        If a vendor delivery is delayed and the product reaches critical risk level (stock < safety stock), the Store Manager must trigger a vendor escalation ticket."""
    },
    {
        "title": "Shift Handover and Cash Audit SOP",
        "category": "Store Operations",
        "content": """Standard Operating Procedure for Shift Handover & Reconciliation:
        At the end of each shift (Morning: 6 AM - 2 PM, Evening: 2 PM - 10 PM, Night: 10 PM - 6 AM):
        1. Complete a POS till cash count. Discrepancies exceeding $5.00 must be reported to the Store Manager.
        2. Verify that all safe drops are logged in the drop safe logs.
        3. Conduct a physical stock count of high-value items (tobacco, phone cards, lottery).
        4. Clean the Wild Bean Cafe coffee machines using the auto-clean cycle.
        5. Log key handovers in the manager's diary."""
    },
    {
        "title": "Fuel Leakage Detection Procedures",
        "category": "Compliance & Safety",
        "content": """Fuel Tank Leakage and Reconciliation Policy:
        Underground storage tanks (USTs) utilize Automatic Tank Gauging (ATG) systems.
        ATG checks for pressure drops and water intrusion in fuel lines.
        Daily Wet Stock Reconciliation: Store Managers must record fuel inventory reconciliation reports daily, comparing ATG readings against pump sales.
        If the Daily Variance exceeds 0.5% of total throughput for 3 consecutive days, a suspected leak alarm is triggered. The manager must contact compliance officers and arrange testing within 24 hours."""
    }
]

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
                    self.db_docs = data["docs"]
                    self.embeddings_matrix = np.array(data["embeddings"])
                return
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
        print(f"Score: {r['score']:.4f} | Title: {r['doc']['title']}")
        print(r['doc']['content'][:200] + "...\n")
