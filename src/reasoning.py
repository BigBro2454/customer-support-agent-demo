import json
from typing import Literal, Any
from pydantic import BaseModel, Field
from google import genai
try:
    from src.config import CONFIG
    from src.perception import MessageIntent
except ModuleNotFoundError:
    from config import CONFIG
    from perception import MessageIntent

class AgentDecision(BaseModel):
    action_type: Literal["resolve", "escalate"] = Field(
        description="Whether the agent can autonomously 'resolve' the issue or needs to 'escalate' to a human."
    )
    confidence: float = Field(
        description="The agent's confidence in this decision, from 0.0 to 1.0."
    )
    rationale: str = Field(
        description="A detailed explanation of why the agent chose to resolve or escalate, referencing policies."
    )
    action_parameters: dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the resolution if applicable (e.g., {'amount': 40.0, 'order_id': '123'})."
    )

class ReasoningModule:
    """
    Reasoning Module represents the core decision-making logic of the agent.
    It applies the retrieved policies to the classified intent to decide whether to act or escalate.
    """
    def __init__(self):
        self.client = genai.Client(api_key=CONFIG["GEMINI_API_KEY"])
        
    def decide_action(self, intent: MessageIntent, retrieved_policies: list[str]) -> AgentDecision:
        """
        Determines the appropriate action based on intent, sentiment, and retrieved policies.
        """
        print(f"🤔 [Reasoning] Evaluating intent against retrieved policies...")
        
        # Combine the policies into a single string for the prompt
        policies_text = "\n\n".join(retrieved_policies)
        
        prompt = f"""
        You are an autonomous customer support agent. 
        Your job is to decide whether you can RESOLVE the customer's request autonomously or if you must ESCALATE it to a human.
        
        You have a STRICT CONFIDENCE THRESHOLD of {CONFIG['CONFIDENCE_THRESHOLD']}.
        If your confidence is below this, or if the action is explicitly deemed high-risk by the escalation rules, you MUST escalate.
        
        --- CUSTOMER CONTEXT ---
        Intent: {intent.intent}
        Urgency: {intent.urgency}
        Sentiment: {intent.sentiment}
        Extracted Entities: {intent.entities.model_dump_json()}
        
        --- RELEVANT COMPANY POLICIES ---
        {policies_text}
        
        --- INSTRUCTIONS ---
        1. Evaluate the customer's request against the provided policies.
        2. Decide if this can be handled autonomously (resolve) or if it requires a human (escalate).
        3. Provide a confidence score (0.0 to 1.0).
        4. Provide a clear rationale referencing the policies.
        5. If resolving, provide any necessary action_parameters (e.g., amount to refund, order_id).
        """
        
        response = self.client.models.generate_content(
            model=CONFIG["MODEL_NAME"],
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": AgentDecision,
                "temperature": 0.2, # Low temperature for more logical reasoning
            },
        )
        
        decision = AgentDecision.model_validate_json(response.text)
        
        # Post-LLM hardcoded safeguards (Defense-in-Depth)
        if decision.action_type == "resolve":
            # Safeguard 1: Enforce confidence threshold programmatically
            if decision.confidence < CONFIG["CONFIDENCE_THRESHOLD"]:
                print(f"⚠️ [Reasoning] OVERRIDE: LLM suggested resolve, but confidence ({decision.confidence:.2f}) is below threshold ({CONFIG['CONFIDENCE_THRESHOLD']}). Forcing escalation.")
                decision.action_type = "escalate"
                decision.rationale = f"Agent override: Confidence ({decision.confidence:.2f}) was below the required threshold of {CONFIG['CONFIDENCE_THRESHOLD']}."
            
            # Safeguard 2: Enforce financial policy boundary programmatically
            amount = intent.entities.amount or decision.action_parameters.get("amount")
            if amount is not None:
                try:
                    numeric_amount = float(amount)
                    if numeric_amount > CONFIG["MAX_REFUND_AUTO"]:
                        print(f"⚠️ [Reasoning] OVERRIDE: Refund amount (${numeric_amount:.2f}) exceeds autonomous limit (${CONFIG['MAX_REFUND_AUTO']:.2f}). Forcing escalation.")
                        decision.action_type = "escalate"
                        decision.rationale = f"Policy override: Refund amount (${numeric_amount:.2f}) exceeds autonomous threshold (${CONFIG['MAX_REFUND_AUTO']:.2f}). Requires supervisor approval."
                except (ValueError, TypeError):
                    pass
            
        return decision
