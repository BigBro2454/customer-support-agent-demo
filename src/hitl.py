"""
Human-in-the-Loop (HITL) Supervisor Escalation Queue & Review Console.
Provides prioritized queueing for human support supervisors, priority scoring,
and programmatic supervisor review workflows (Approve, Override, Reject).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class EscalatedTicket(BaseModel):
    ticket_id: str
    created_at: str
    priority: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    customer_message: str
    intent: str
    sentiment: str
    urgency: str
    escalation_reason: str
    retrieved_policies: list[str] = Field(default_factory=list)
    suggested_action: str = Field(default="manual_review")
    status: Literal["PENDING", "APPROVED", "OVERRIDDEN", "REJECTED"] = "PENDING"
    supervisor_notes: str | None = None
    resolved_at: str | None = None
    financial_amount: float | None = None


class SupervisorQueue:
    """
    Priority-ordered queue for Human-in-the-Loop ticket management.
    """

    PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    def __init__(self, storage_dir: str | None = None):
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.queue: list[EscalatedTicket] = []

    def calculate_priority(
        self,
        sentiment: str,
        escalation_reason: str,
        amount: float | None = None,
        detected_threats: list[str] | None = None,
    ) -> Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        """
        Calculates priority score based on risk vectors and financial boundaries.
        """
        threats = detected_threats or []

        # 1. Critical priority: Security threats, prompt injections, legal threats
        if "PROMPT_INJECTION_ATTEMPT" in threats or "LEGAL_ESCALATION_THREAT" in threats:
            return "CRITICAL"
        if "lawyer" in escalation_reason.lower() or "attorney" in escalation_reason.lower():
            return "CRITICAL"
        if amount is not None and amount >= 200.0:
            return "CRITICAL"

        # 2. High priority: Exceeding auto limit ($50-$200) or angry sentiment
        if amount is not None and amount > 50.0:
            return "HIGH"
        if sentiment in ("angry", "hostile", "negative"):
            return "HIGH"

        # 3. Medium priority: Policy ambiguity, low confidence
        if "confidence" in escalation_reason.lower() or "threshold" in escalation_reason.lower():
            return "MEDIUM"

        return "LOW"

    def enqueue(
        self,
        ticket_id: str,
        customer_message: str,
        intent: str,
        sentiment: str,
        urgency: str,
        escalation_reason: str,
        retrieved_policies: list[str] | None = None,
        suggested_action: str = "manual_review",
        amount: float | None = None,
        detected_threats: list[str] | None = None,
    ) -> EscalatedTicket:
        """Enqueues a ticket into the supervisor escalation queue with calculated priority."""
        priority = self.calculate_priority(sentiment, escalation_reason, amount, detected_threats)
        ticket = EscalatedTicket(
            ticket_id=ticket_id,
            created_at=datetime.now().isoformat(),
            priority=priority,
            customer_message=customer_message,
            intent=intent,
            sentiment=sentiment,
            urgency=urgency,
            escalation_reason=escalation_reason,
            retrieved_policies=retrieved_policies or [],
            suggested_action=suggested_action,
            financial_amount=amount,
        )
        self.queue.append(ticket)
        self._save_queue()
        return ticket

    def get_pending(self, priority_filter: str | None = None) -> list[EscalatedTicket]:
        """Returns pending tickets ordered by priority (CRITICAL -> HIGH -> MEDIUM -> LOW)."""
        pending = [t for t in self.queue if t.status == "PENDING"]
        if priority_filter:
            pending = [t for t in pending if t.priority == priority_filter.upper()]

        return sorted(pending, key=lambda t: self.PRIORITY_ORDER.get(t.priority, 99))

    def review_ticket(
        self,
        ticket_id: str,
        action: Literal["APPROVED", "OVERRIDDEN", "REJECTED"],
        supervisor_notes: str,
    ) -> EscalatedTicket | None:
        """Processes a supervisor review action on an escalated ticket."""
        for ticket in self.queue:
            if ticket.ticket_id == ticket_id:
                ticket.status = action
                ticket.supervisor_notes = supervisor_notes
                ticket.resolved_at = datetime.now().isoformat()
                self._save_queue()
                return ticket
        return None

    def get_stats(self) -> dict:
        """Returns queue statistics breakdown."""
        total = len(self.queue)
        pending = sum(1 for t in self.queue if t.status == "PENDING")
        approved = sum(1 for t in self.queue if t.status == "APPROVED")
        overridden = sum(1 for t in self.queue if t.status == "OVERRIDDEN")
        rejected = sum(1 for t in self.queue if t.status == "REJECTED")

        critical = sum(1 for t in self.queue if t.priority == "CRITICAL" and t.status == "PENDING")
        high = sum(1 for t in self.queue if t.priority == "HIGH" and t.status == "PENDING")

        return {
            "total_escalations": total,
            "pending_count": pending,
            "critical_pending": critical,
            "high_pending": high,
            "approved_count": approved,
            "overridden_count": overridden,
            "rejected_count": rejected,
        }

    def _save_queue(self):
        """Persists the queue state to disk."""
        filepath = os.path.join(self.storage_dir, "hitl_escalation_queue.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in self.queue], f, indent=2)
        except Exception:
            pass
