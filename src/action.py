import os
import json
import datetime
try:
    from src.perception import MessageIntent
    from src.reasoning import AgentDecision
except ModuleNotFoundError:
    from perception import MessageIntent
    from reasoning import AgentDecision

class ActionModule:
    """
    Action Module executes the final decision made by the reasoning module.
    It simulates calling external APIs and logs an audit trail.
    """
    def __init__(self, log_dir: str | None = None):
        self.audit_log = []
        if log_dir:
            self.log_dir = log_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def execute(self, original_message: str, intent: MessageIntent, decision: AgentDecision):
        """
        Executes the decision and logs the result.
        """
        print(f"\n⚡ [Action] Executing decision: {decision.action_type.upper()}")
        print(f"   Rationale: {decision.rationale}")
        print(f"   Confidence: {decision.confidence:.2f}")
        
        if decision.action_type == "resolve":
            self._handle_resolve(intent, decision)
        else:
            self._handle_escalate(original_message, intent, decision)
            
        self._log_audit(original_message, intent, decision)
            
    def _handle_resolve(self, intent: MessageIntent, decision: AgentDecision):
        """Mock execution of a resolution"""
        print("\n✅ --- RESOLUTION EXECUTED ---")
        if intent.intent == "refund":
            amount = decision.action_parameters.get("amount", "unknown")
            order_id = decision.action_parameters.get("order_id", "unknown")
            print(f"💰 Processing refund of ${amount} for order {order_id} via Stripe API...")
        elif intent.intent == "password_reset":
            email = decision.action_parameters.get("email", intent.entities.email or "unknown")
            print(f"📧 Sending password reset link to {email} via SendGrid API...")
        else:
            print(f"💬 Providing automated response for intent: {intent.intent}")
        print("------------------------------\n")

    def _handle_escalate(self, original_message: str, intent: MessageIntent, decision: AgentDecision):
        """Mock execution of an escalation to a human agent"""
        print("\n🚨 --- TICKET ESCALATED TO HUMAN ---")
        print("Creating Zendesk ticket with following context:")
        print(f"Message: '{original_message}'")
        print(f"Classified Intent: {intent.intent} | Sentiment: {intent.sentiment}")
        print(f"Agent Rationale for Escalation: {decision.rationale}")
        print("------------------------------------\n")

    def _log_audit(self, original_message: str, intent: MessageIntent, decision: AgentDecision):
        """Logs every action to an audit trail (episodic memory)."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": original_message,
            "parsed_intent": intent.model_dump(),
            "decision": decision.model_dump()
        }
        self.audit_log.append(log_entry)
        
        # Persist structured JSONL log
        log_file = os.path.join(self.log_dir, "audit_trail.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
