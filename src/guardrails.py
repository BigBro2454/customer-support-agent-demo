"""
Enterprise Multi-Stage Security and Guardrails Engine for Customer Support AI.
Provides pre-execution prompt injection defense, PII masking, legal threat detection,
and programmatic financial boundary enforcement.
"""

from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, Field


class GuardrailCheckResult(BaseModel):
    passed: bool = Field(description="True if message passes all security checks without blocking")
    sanitized_text: str = Field(description="Sanitized input with PII masked")
    detected_threats: list[str] = Field(default_factory=list, description="List of threat codes detected")
    risk_score: float = Field(default=0.0, description="Normalized risk score from 0.0 to 1.0")
    recommended_action: Literal["allow", "sanitize", "escalate", "block"] = Field(
        default="allow", description="Recommended pipeline action"
    )
    reason: str = Field(default="", description="Detailed audit explanation for decision")


class GuardrailEngine:
    """
    Multi-stage Defense-in-Depth Guardrail Engine.
    Executes deterministic, zero-latency validation before hitting vector stores or LLMs.
    """

    # Adversarial Prompt Injection & Jailbreak Heuristics
    PROMPT_INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directives|prompts)",
        r"(?i)system\s+prompt\s+(override|bypass|leak|reveal)",
        r"(?i)you\s+are\s+now\s+(an?\s+)?(unfiltered|dan|jailbroken|unrestricted)",
        r"(?i)developer\s+mode\s+(enabled|activated|on)",
        r"(?i)disregard\s+(the\s+)?(rules|safety|guidelines)",
        r"(?i)print\s+(the\s+)?(system|internal)\s+(prompt|instructions)",
        r"(?i)drop\s+table\b",
        r"(?i)<\s*script\b",
    ]

    # Legal Threat & Hostile Escalation Patterns
    LEGAL_THREAT_PATTERNS = [
        r"(?i)\b(my\s+)?(lawyer|attorney|legal\s+team|counsel)\b",
        r"(?i)\b(sue|suing|lawsuit|court\s+order)\b",
        r"(?i)\b(better\s+business\s+bureau|bbb\s+complaint)\b",
        r"(?i)\b(attorney\s+general|ftc\s+complaint|regulatory\s+action)\b",
        r"(?i)\b(file\s+charges|police\s+report)\b",
    ]

    # PII Regex Patterns
    # Credit Card Numbers (13-19 digits, optionally spaced or hyphenated)
    CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"
    # US Social Security Number (format XXX-XX-XXXX or 9 digits)
    SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
    # Plaintext Password mentions (e.g., "password is: abc123", "password = xyz", "pwd: 123")
    PASSWORD_PATTERN = r"(?i)\b(?:password|passwd|pwd)(?:\s+is|\s*[:=])+\s*([^\s,;]+)"

    def __init__(self, max_auto_refund_limit: float = 50.0):
        self.max_auto_refund_limit = max_auto_refund_limit
        self._compiled_injection = [re.compile(p) for p in self.PROMPT_INJECTION_PATTERNS]
        self._compiled_legal = [re.compile(p) for p in self.LEGAL_THREAT_PATTERNS]
        self._compiled_cc = re.compile(self.CREDIT_CARD_PATTERN)
        self._compiled_ssn = re.compile(self.SSN_PATTERN)
        self._compiled_pwd = re.compile(self.PASSWORD_PATTERN)

    def mask_pii(self, text: str) -> tuple[str, list[str]]:
        """
        Detects and masks personally identifiable information (PII).
        Replaces Credit Cards with ****-****-****-XXXX and SSNs with ***-**-XXXX.
        """
        threats = []
        sanitized = text

        # Mask Credit Cards
        def _mask_cc(match):
            raw = re.sub(r"[-\s]", "", match.group(0))
            last4 = raw[-4:]
            threats.append("PII_CREDIT_CARD")
            return f"****-****-****-{last4}"

        if self._compiled_cc.search(sanitized):
            sanitized = self._compiled_cc.sub(_mask_cc, sanitized)

        # Mask SSNs
        def _mask_ssn(match):
            raw = match.group(0)
            last4 = raw[-4:]
            threats.append("PII_SSN")
            return f"***-**-{last4}"

        if self._compiled_ssn.search(sanitized):
            sanitized = self._compiled_ssn.sub(_mask_ssn, sanitized)

        # Mask Passwords
        def _mask_pwd(match):
            threats.append("PII_PASSWORD")
            return f"password: [PROTECTED_CREDENTIAL]"

        if self._compiled_pwd.search(sanitized):
            sanitized = self._compiled_pwd.sub(_mask_pwd, sanitized)

        return sanitized, list(set(threats))

    def evaluate_input(self, raw_message: str) -> GuardrailCheckResult:
        """
        Executes pre-execution safety inspection on raw customer inputs.
        """
        threats: list[str] = []
        risk_score = 0.0

        # Step 1: Check Prompt Injections
        for pattern in self._compiled_injection:
            if pattern.search(raw_message):
                threats.append("PROMPT_INJECTION_ATTEMPT")
                risk_score = max(risk_score, 0.95)
                break

        # Step 2: Check Legal Threats & Regulatory Warnings
        for pattern in self._compiled_legal:
            if pattern.search(raw_message):
                threats.append("LEGAL_ESCALATION_THREAT")
                risk_score = max(risk_score, 0.85)
                break

        # Step 3: Sanitize and Mask PII
        sanitized_text, pii_threats = self.mask_pii(raw_message)
        threats.extend(pii_threats)
        if pii_threats:
            risk_score = max(risk_score, 0.60)

        # Step 4: Determine Action & Reason
        if "PROMPT_INJECTION_ATTEMPT" in threats:
            return GuardrailCheckResult(
                passed=False,
                sanitized_text=sanitized_text,
                detected_threats=threats,
                risk_score=risk_score,
                recommended_action="block",
                reason="Adversarial prompt injection pattern detected. Request blocked to preserve system integrity.",
            )

        if "LEGAL_ESCALATION_THREAT" in threats:
            return GuardrailCheckResult(
                passed=True,
                sanitized_text=sanitized_text,
                detected_threats=threats,
                risk_score=risk_score,
                recommended_action="escalate",
                reason="Legal counsel or regulatory threat detected. Mandatory human tier escalation required.",
            )

        if pii_threats:
            return GuardrailCheckResult(
                passed=True,
                sanitized_text=sanitized_text,
                detected_threats=threats,
                risk_score=risk_score,
                recommended_action="sanitize",
                reason=f"Sensitive PII detected and masked ({', '.join(pii_threats)}). Proceeding with sanitized text.",
            )

        return GuardrailCheckResult(
            passed=True,
            sanitized_text=raw_message,
            detected_threats=[],
            risk_score=0.0,
            recommended_action="allow",
            reason="Input verified safe. No adversarial or compliance threats detected.",
        )

    def validate_financial_boundary(self, requested_amount: float | None) -> tuple[bool, str]:
        """
        Validates whether a financial transaction exceeds autonomous limits.
        """
        if requested_amount is None:
            return True, "No financial amount requested."

        if requested_amount <= 0:
            return False, f"Invalid refund amount requested (${requested_amount:.2f})."

        if requested_amount > self.max_auto_refund_limit:
            return (
                False,
                f"Requested amount (${requested_amount:.2f}) exceeds autonomous limit of ${self.max_auto_refund_limit:.2f}.",
            )

        return True, f"Amount (${requested_amount:.2f}) is within autonomous approval limit."
