#!/usr/bin/env python3
"""Automated Incident Triage, Semantic Deduplication & Root Cause Classifier.

Demonstrates Google L5 Support Systems Engineering:
1. Emerging Outage & Anomaly Detection (clusters similar incoming support tickets).
2. Root Cause Classification (Infrastructure, Billing Gateway, Auth, App Defect, User Config).
3. Automated Mass Broadcast Generation (mitigates ticket storms with instant customer notifications).
4. Semantic Deduplication & Incident Blast Radius Accounting.
"""

import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RootCauseCategory(str, Enum):
    INFRASTRUCTURE_OUTAGE = "INFRASTRUCTURE_OUTAGE"
    BILLING_GATEWAY = "BILLING_GATEWAY"
    AUTH_CREDENTIAL = "AUTH_CREDENTIAL"
    APPLICATION_DEFECT = "APPLICATION_DEFECT"
    USER_CONFIGURATION = "USER_CONFIGURATION"


class IncidentSeverity(str, Enum):
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"


@dataclass
class SupportIncident:
    """Represents an ongoing aggregated support incident."""
    incident_id: str
    title: str
    root_cause: str
    severity: str
    ticket_count: int
    ticket_ids: List[str]
    sample_queries: List[str]
    affected_components: List[str]
    broadcast_notification: str
    status: str = "ACTIVE"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IncidentClusterer:
    """Real-time ticket clustering, anomaly grouping, and root cause classification engine."""

    # Heuristic signature rules for root cause identification
    ROOT_CAUSE_SIGNATURES = [
        (
            re.compile(r"(?i)\b(?:502|503|504|gateway\s+timeout|internal\s+server\s+error|service\s+unavailable|downtime|outage|unreachable|server\s+crash)\b"),
            RootCauseCategory.INFRASTRUCTURE_OUTAGE,
            IncidentSeverity.P0_CRITICAL,
            ["API Gateway", "Core Kubernetes Cluster"],
        ),
        (
            re.compile(r"(?i)\b(?:payment\s+declined|card\s+error|stripe|billing\s+failed|double\s+charge|duplicate\s+charge|checkout\s+crash|invoice\s+error|refund\s+failure)\b"),
            RootCauseCategory.BILLING_GATEWAY,
            IncidentSeverity.P1_HIGH,
            ["Stripe Gateway", "Billing Ledger Service"],
        ),
        (
            re.compile(r"(?i)\b(?:sso|saml|oauth|login\s+loop|token\s+expired|invalid\s+credentials|2fa|mfa|password\s+reset|forbidden|403|unauthorized)\b"),
            RootCauseCategory.AUTH_CREDENTIAL,
            IncidentSeverity.P1_HIGH,
            ["Identity Provider", "OAuth2 Token Service"],
        ),
        (
            re.compile(r"(?i)\b(?:null\s+pointer|typeerror|uncaught\s+exception|blank\s+screen|ui\s+freeze|export\s+fails|button\s+not\s+working|broken\s+link|javascript\s+error)\b"),
            RootCauseCategory.APPLICATION_DEFECT,
            IncidentSeverity.P2_MEDIUM,
            ["Web Dashboard", "Export Microservice"],
        ),
        (
            re.compile(r"(?i)\b(?:how\s+do\s+i|where\s+is|configure|setup|firewall|whitelist|ip\s+address|documentation|settings)\b"),
            RootCauseCategory.USER_CONFIGURATION,
            IncidentSeverity.P3_LOW,
            ["Customer Environment", "Self-Serve Portal"],
        ),
    ]

    def __init__(self, cluster_threshold: float = 0.25):
        """Initialize incident clusterer.

        Args:
            cluster_threshold: Jaccard keyword similarity threshold to merge tickets into an incident.
        """
        self.cluster_threshold = cluster_threshold
        self.active_incidents: Dict[str, SupportIncident] = {}
        self.ticket_to_incident_map: Dict[str, str] = {}
        self.total_tickets_processed: int = 0
        self._incident_counter: int = 0

    def _tokenize(self, text: str) -> set:
        """Tokenize text into lowercased semantic keyword set."""
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        # Filter common stopwords
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "have", "not", "are", "can", "our", "all"}
        return {w for w in words if w not in stopwords}

    def _calculate_similarity(self, tokens_a: set, tokens_b: set) -> float:
        """Compute Jaccard token similarity coefficient."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    def classify_root_cause(self, query: str) -> Tuple[str, str, List[str]]:
        """Classify root cause, severity, and impacted components based on query content."""
        for pattern, category, severity, components in self.ROOT_CAUSE_SIGNATURES:
            if pattern.search(query):
                return category.value, severity.value, components

        # Default fallback
        return (
            RootCauseCategory.APPLICATION_DEFECT.value,
            IncidentSeverity.P2_MEDIUM.value,
            ["General Application Service"],
        )

    def _generate_broadcast_message(self, root_cause: str, title: str, components: List[str]) -> str:
        """Synthesize customer-facing broadcast notification banner."""
        comp_str = ", ".join(components)
        if root_cause == RootCauseCategory.INFRASTRUCTURE_OUTAGE.value:
            return (
                f"🚨 [STATUS ADVISORY] We are actively investigating elevated errors on {comp_str}. "
                "Our SRE team is working to restore normal operations. Ticket updates will be broadcast automatically."
            )
        elif root_cause == RootCauseCategory.BILLING_GATEWAY.value:
            return (
                f"⚠️ [BILLING ADVISORY] We are currently experiencing latency with payment processing on {comp_str}. "
                "Please do not submit duplicate checkout attempts while transactions are being reconciled."
            )
        elif root_cause == RootCauseCategory.AUTH_CREDENTIAL.value:
            return (
                f"🔒 [AUTHENTICATION NOTICE] Some users may experience intermittent login delays on {comp_str}. "
                "Existing active sessions are unaffected. Token refresh fixes are being deployed."
            )
        else:
            return (
                f"ℹ️ [SERVICE UPDATE] Elevated inquiries detected regarding {title}. "
                "Our team is monitoring system telemetry to ensure uninterrupted service."
            )

    def process_ticket(
        self, ticket_id: str, query: str, customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Triage an incoming support ticket, cluster against existing incidents, or create a new incident.

        Returns:
            Dictionary containing triage resolution, linked incident ID, and broadcast advisories.
        """
        self.total_tickets_processed += 1
        query_tokens = self._tokenize(query)
        best_match_incident: Optional[SupportIncident] = None
        highest_similarity = 0.0

        query_rc, query_sev, query_comps = self.classify_root_cause(query)

        # Scan active incidents for semantic clustering
        for incident in self.active_incidents.values():
            for sample in incident.sample_queries:
                sample_tokens = self._tokenize(sample)
                sim = self._calculate_similarity(query_tokens, sample_tokens)
                # Boost similarity if matching the exact same root cause category
                if incident.root_cause == query_rc:
                    sim = min(1.0, sim + 0.15)
                if sim > highest_similarity:
                    highest_similarity = sim
                    best_match_incident = incident

        # If similarity meets threshold, merge into existing incident
        if best_match_incident and highest_similarity >= self.cluster_threshold:
            best_match_incident.ticket_count += 1
            best_match_incident.ticket_ids.append(ticket_id)
            if len(best_match_incident.sample_queries) < 5:
                best_match_incident.sample_queries.append(query)
            best_match_incident.last_seen = time.time()
            self.ticket_to_incident_map[ticket_id] = best_match_incident.incident_id

            return {
                "ticket_id": ticket_id,
                "is_clustered": True,
                "incident_id": best_match_incident.incident_id,
                "incident_title": best_match_incident.title,
                "severity": best_match_incident.severity,
                "root_cause": best_match_incident.root_cause,
                "similarity_score": round(highest_similarity, 3),
                "broadcast_banner": best_match_incident.broadcast_notification,
                "cluster_ticket_count": best_match_incident.ticket_count,
            }

        # Otherwise, create a new incident track
        self._incident_counter += 1
        inc_id = f"INC-{int(time.time() * 1000) % 100000:05d}-{self._incident_counter:02d}"
        
        # Extract title from query
        cleaned_words = [w for w in query.split() if len(w) > 3][:6]
        title = " ".join(cleaned_words).capitalize() or "System Anomaly"

        broadcast = self._generate_broadcast_message(query_rc, title, query_comps)

        new_incident = SupportIncident(
            incident_id=inc_id,
            title=title,
            root_cause=query_rc,
            severity=query_sev,
            ticket_count=1,
            ticket_ids=[ticket_id],
            sample_queries=[query],
            affected_components=query_comps,
            broadcast_notification=broadcast,
        )

        self.active_incidents[inc_id] = new_incident
        self.ticket_to_incident_map[ticket_id] = inc_id


        return {
            "ticket_id": ticket_id,
            "is_clustered": False,
            "incident_id": inc_id,
            "incident_title": title,
            "severity": query_sev,
            "root_cause": query_rc,
            "similarity_score": 1.0,
            "broadcast_banner": broadcast,
            "cluster_ticket_count": 1,
        }

    def get_incident_summary(self) -> Dict[str, Any]:
        """Generate high-level triage summary across all active incidents."""
        incidents_list = [inc.to_dict() for inc in self.active_incidents.values()]
        total_incidents = len(incidents_list)
        total_clustered_tickets = sum(inc["ticket_count"] for inc in incidents_list)

        severity_counts = {s.value: 0 for s in IncidentSeverity}
        root_cause_counts = {r.value: 0 for r in RootCauseCategory}

        for inc in incidents_list:
            severity_counts[inc["severity"]] = severity_counts.get(inc["severity"], 0) + 1
            root_cause_counts[inc["root_cause"]] = root_cause_counts.get(inc["root_cause"], 0) + 1

        return {
            "total_tickets_processed": self.total_tickets_processed,
            "total_active_incidents": total_incidents,
            "total_clustered_tickets": total_clustered_tickets,
            "severity_distribution": severity_counts,
            "root_cause_distribution": root_cause_counts,
            "incidents": incidents_list,
        }


# Singleton instance
_GLOBAL_CLUSTERER = IncidentClusterer()


def get_incident_clusterer() -> IncidentClusterer:
    """Return singleton clusterer engine."""
    return _GLOBAL_CLUSTERER
