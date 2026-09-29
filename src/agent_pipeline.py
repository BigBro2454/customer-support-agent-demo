"""
Unified Customer Support Agent Pipeline with Defense-in-Depth, HITL, and Telemetry.
Combines Pre-execution Security, Semantic Perception, Vector RAG Retrieval,
Policy-bounded LLM Reasoning, Human-in-the-Loop Queueing, and Execution Telemetry.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any
from pydantic import BaseModel, Field

try:
    from src.config import CONFIG
    from src.guardrails import GuardrailEngine, GuardrailCheckResult
    from src.perception import PerceptionModule, MessageIntent, ExtractedEntities
    from src.memory import MemoryModule
    from src.reasoning import ReasoningModule, AgentDecision
    from src.action import ActionModule
    from src.hitl import SupervisorQueue, EscalatedTicket
    from src.telemetry import SessionTelemetryCollector, LatencyBreakdown, TicketTelemetry
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import CONFIG
    from src.guardrails import GuardrailEngine, GuardrailCheckResult
    from src.perception import PerceptionModule, MessageIntent, ExtractedEntities
    from src.memory import MemoryModule
    from src.reasoning import ReasoningModule, AgentDecision
    from src.action import ActionModule
    from src.hitl import SupervisorQueue, EscalatedTicket
    from src.telemetry import SessionTelemetryCollector, LatencyBreakdown, TicketTelemetry


class PipelineExecutionResult(BaseModel):
    ticket_id: str
    original_message: str
    sanitized_message: str
    guardrail: GuardrailCheckResult
    intent: MessageIntent | None = None
    retrieved_policies: list[str] = Field(default_factory=list)
    decision: AgentDecision | None = None
    escalated_ticket: EscalatedTicket | None = None
    telemetry: TicketTelemetry


class CustomerSupportPipeline:
    """
    Production-grade Customer Support Agent Orchestrator.
    Supports both live (Gemini 2.5 Flash + ChromaDB) and offline mock execution.
    """

    def __init__(
        self,
        mock_mode: bool = False,
        log_dir: str | None = None,
        chroma_persist_dir: str | None = None,
    ):
        self.mock_mode = mock_mode
        self.guardrail_engine = GuardrailEngine(max_auto_refund_limit=CONFIG["MAX_REFUND_AUTO"])
        self.supervisor_queue = SupervisorQueue(storage_dir=log_dir)
        self.telemetry_collector = SessionTelemetryCollector(log_dir=log_dir)
        self.action_module = ActionModule(log_dir=log_dir)

        # Initialize Memory (ChromaDB)
        try:
            self.memory_module = MemoryModule(persist_dir=chroma_persist_dir)
        except Exception as e:
            if not self.mock_mode:
                raise e
            self.memory_module = None

        # Initialize Perception & Reasoning (Requires GEMINI_API_KEY for live)
        if not self.mock_mode:
            self.perception_module = PerceptionModule()
            self.reasoning_module = ReasoningModule()
        else:
            self.perception_module = None
            self.reasoning_module = None

    def _mock_perception(self, message: str) -> MessageIntent:
        """Deterministic offline rule-based perception for CI/CD runs."""
        msg_lower = message.lower()

        # Order ID extraction
        order_match = re.search(r"#?(\b\d{4,6}\b)", message)
        order_id = f"#{order_match.group(1)}" if order_match else None

        # Amount extraction
        amount_match = re.search(r"\$(\d[\d,]*(?:\.\d{2})?)", message)
        amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

        # Email extraction
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", message)
        email = email_match.group(0) if email_match else None

        # Intent classification
        if any(w in msg_lower for w in ["refund", "money back", "return", "charge"]):
            intent = "refund"
        elif any(w in msg_lower for w in ["password", "login", "reset", "access"]):
            intent = "password_reset"
        elif any(w in msg_lower for w in ["credit card", "billing", "invoice", "payment", "update"]):
            intent = "billing_question"
        elif any(w in msg_lower for w in ["delete", "gdpr", "remove data", "cancel account"]):
            intent = "account_deletion"
        else:
            intent = "general_support"

        urgency = "high" if any(w in msg_lower for w in ["immediately", "urgent", "right now", "terrible", "demand"]) else "medium"
        sentiment = "angry" if any(w in msg_lower for w in ["terrible", "awful", "scam", "demand", "hate", "lawsuit"]) else "neutral"

        return MessageIntent(
            intent=intent,
            urgency=urgency,
            sentiment=sentiment,
            entities=ExtractedEntities(order_id=order_id, amount=amount, email=email),
        )

    def _mock_reasoning(self, intent: MessageIntent, policies: list[str]) -> AgentDecision:
        """Deterministic offline reasoning logic adhering to company policy."""
        # Policy boundaries
        if intent.intent == "refund":
            amt = intent.entities.amount
            if amt is not None and amt > CONFIG["MAX_REFUND_AUTO"]:
                return AgentDecision(
                    action_type="escalate",
                    confidence=0.98,
                    rationale=f"Policy override: Refund amount (${amt:.2f}) exceeds autonomous limit of ${CONFIG['MAX_REFUND_AUTO']:.2f}. Requires supervisor sign-off.",
                    action_parameters={"order_id": intent.entities.order_id, "amount": amt},
                )
            if intent.sentiment == "angry" and (amt and amt > 30.0):
                return AgentDecision(
                    action_type="escalate",
                    confidence=0.88,
                    rationale="Customer sentiment is highly negative with significant refund demand. Escalating to prevent churn.",
                    action_parameters={"order_id": intent.entities.order_id, "amount": amt},
                )
            return AgentDecision(
                action_type="resolve",
                confidence=0.95,
                rationale=f"Refund request of ${amt or 25.0:.2f} is within autonomous approval limits and customer is within 30-day window.",
                action_parameters={"order_id": intent.entities.order_id or "#9999", "amount": amt or 25.0},
            )

        if intent.intent == "password_reset":
            return AgentDecision(
                action_type="resolve",
                confidence=0.99,
                rationale="Automated identity verification link sent to customer email on record.",
                action_parameters={"email": intent.entities.email or "customer@example.com"},
            )

        if intent.intent == "account_deletion":
            return AgentDecision(
                action_type="escalate",
                confidence=0.99,
                rationale="GDPR / California Privacy Rights account deletion requires manual identity proof and legal verification.",
                action_parameters={},
            )

        if intent.intent == "billing_question":
            return AgentDecision(
                action_type="resolve",
                confidence=0.92,
                rationale="Guided customer to billing settings portal via standard self-service workflow.",
                action_parameters={},
            )

        return AgentDecision(
            action_type="escalate",
            confidence=0.55,
            rationale="Unrecognized customer intent or ambiguous policy. Escalating for manual review.",
            action_parameters={},
        )

    def process_ticket(self, raw_message: str, ticket_id: str | None = None) -> PipelineExecutionResult:
        """
        Executes end-to-end processing of a customer support ticket.
        """
        if not ticket_id:
            ticket_id = f"TICK-{int(time.time() * 1000) % 1_000_000}"

        t0 = time.perf_counter()

        # Phase 1: Security & Guardrails Pre-Check
        t_g0 = time.perf_counter()
        guardrail_result = self.guardrail_engine.evaluate_input(raw_message)
        guardrail_ms = round((time.perf_counter() - t_g0) * 1000, 2)

        # Immediate Block for Prompt Injections
        if guardrail_result.recommended_action == "block":
            t_total = round((time.perf_counter() - t0) * 1000, 2)
            latency = LatencyBreakdown(guardrail_ms=guardrail_ms, total_ms=t_total)
            telemetry = self.telemetry_collector.record_ticket(
                ticket_id=ticket_id,
                message=raw_message,
                intent="security_blocked",
                urgency="high",
                sentiment="adversarial",
                action_type="block",
                confidence=1.0,
                retrieved_policies_count=0,
                guardrail_flags=guardrail_result.detected_threats,
                latency=latency,
            )
            return PipelineExecutionResult(
                ticket_id=ticket_id,
                original_message=raw_message,
                sanitized_message=guardrail_result.sanitized_text,
                guardrail=guardrail_result,
                telemetry=telemetry,
            )

        # Phase 2: Perception (Parsing & Entity Extraction)
        t_p0 = time.perf_counter()
        if self.mock_mode:
            intent = self._mock_perception(guardrail_result.sanitized_text)
        else:
            intent = self.perception_module.parse_message(guardrail_result.sanitized_text)
        perception_ms = round((time.perf_counter() - t_p0) * 1000, 2)

        # Phase 3: Memory & RAG Retrieval
        t_r0 = time.perf_counter()
        retrieval_query = f"{intent.intent} policy and escalation rules"
        if self.memory_module:
            policies = self.memory_module.retrieve(retrieval_query, n_results=2)
        else:
            policies = [
                "Refund Policy: Orders within 30 days under $50 can be automatically resolved. Over $50 requires escalation.",
                "Escalation Rules: Hostile sentiment, chargeback threats, and legal counsel requests must escalate to Tier 2."
            ]
        retrieval_ms = round((time.perf_counter() - t_r0) * 1000, 2)

        # Phase 4: Reasoning (Policy evaluation & defense-in-depth checks)
        t_rea0 = time.perf_counter()
        if self.mock_mode:
            decision = self._mock_reasoning(intent, policies)
        else:
            decision = self.reasoning_module.decide_action(intent, policies)

        # Defense-in-depth override: If guardrails flagged legal escalation
        if guardrail_result.recommended_action == "escalate" and decision.action_type != "escalate":
            decision.action_type = "escalate"
            decision.rationale = f"Guardrail override: {guardrail_result.reason}"

        reasoning_ms = round((time.perf_counter() - t_rea0) * 1000, 2)

        # Phase 5: Action & HITL Escalation Queueing
        t_a0 = time.perf_counter()
        escalated_ticket = None
        if decision.action_type == "escalate":
            escalated_ticket = self.supervisor_queue.enqueue(
                ticket_id=ticket_id,
                customer_message=guardrail_result.sanitized_text,
                intent=intent.intent,
                sentiment=intent.sentiment,
                urgency=intent.urgency,
                escalation_reason=decision.rationale,
                retrieved_policies=policies,
                amount=intent.entities.amount,
                detected_threats=guardrail_result.detected_threats,
            )

        self.action_module.execute(guardrail_result.sanitized_text, intent, decision)
        action_ms = round((time.perf_counter() - t_a0) * 1000, 2)

        # Total latency and telemetry capture
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency = LatencyBreakdown(
            guardrail_ms=guardrail_ms,
            perception_ms=perception_ms,
            retrieval_ms=retrieval_ms,
            reasoning_ms=reasoning_ms,
            action_ms=action_ms,
            total_ms=total_ms,
        )

        telemetry = self.telemetry_collector.record_ticket(
            ticket_id=ticket_id,
            message=raw_message,
            intent=intent.intent,
            urgency=intent.urgency,
            sentiment=intent.sentiment,
            action_type=decision.action_type,
            confidence=decision.confidence,
            retrieved_policies_count=len(policies),
            guardrail_flags=guardrail_result.detected_threats,
            latency=latency,
        )

        return PipelineExecutionResult(
            ticket_id=ticket_id,
            original_message=raw_message,
            sanitized_message=guardrail_result.sanitized_text,
            guardrail=guardrail_result,
            intent=intent,
            retrieved_policies=policies,
            decision=decision,
            escalated_ticket=escalated_ticket,
            telemetry=telemetry,
        )
