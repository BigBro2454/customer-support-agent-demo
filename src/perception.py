import os
from pydantic import BaseModel, Field
from google import genai
try:
    from src.config import CONFIG
except ModuleNotFoundError:
    from config import CONFIG

class ExtractedEntities(BaseModel):
    order_id: str | None = Field(default=None, description="The order ID mentioned in the message")
    amount: float | None = Field(default=None, description="The refund amount requested, if any")
    email: str | None = Field(default=None, description="The customer's email address, if provided")

class MessageIntent(BaseModel):
    intent: str = Field(description="The primary intent of the customer (e.g., refund, password_reset, billing_question, unknown)")
    urgency: str = Field(description="The urgency of the request (low, medium, high)")
    sentiment: str = Field(description="The sentiment of the customer (positive, neutral, negative, angry)")
    entities: ExtractedEntities = Field(description="Extracted entities from the message")

class PerceptionModule:
    """
    Perception Module parses the customer message to understand what they want.
    """
    def __init__(self):
        # Initialize the Gemini client
        self.client = genai.Client(api_key=CONFIG["GEMINI_API_KEY"])
        
    def parse_message(self, message: str) -> MessageIntent:
        """
        Parses a customer message and extracts intent, urgency, sentiment, and entities.
        """
        print(f"🧠 [Perception] Parsing customer message...")
        
        prompt = f"""
        You are an expert customer support intent classifier.
        Analyze the following customer message and extract the intent, urgency, sentiment, and any relevant entities.
        
        Customer Message: "{message}"
        """
        
        # We use Structured Outputs via Pydantic to ensure we get a clean JSON response
        response = self.client.models.generate_content(
            model=CONFIG["MODEL_NAME"],
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": MessageIntent,
                "temperature": 0.1, # Low temperature for more deterministic classification
            },
        )
        
        # Pydantic validates and parses the JSON response
        return MessageIntent.model_validate_json(response.text)
