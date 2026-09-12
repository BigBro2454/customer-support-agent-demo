import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
CONFIG = {
    # The API key for Google Gemini
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    
    # Model to use for intent parsing and reasoning
    "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-2.5-flash"),
    
    # Threshold for confidence level (below this, the agent should escalate)
    "CONFIDENCE_THRESHOLD": float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
    
    # High risk action thresholds
    "MAX_REFUND_AUTO": float(os.getenv("MAX_REFUND_AUTO", "50.0")),
    "MAX_REFUND_APPROVAL": float(os.getenv("MAX_REFUND_APPROVAL", "200.0")),
    
    # Path to the knowledge base directory
    "KB_DIR": os.getenv("KB_DIR", "knowledge-base"),
    
    # ChromaDB collection name
    "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION", "support_kb")
}
