#!/usr/bin/env python3
"""Automated Unit Tests for Incident Triage & Model Economics Benchmark Suite.

Validates:
1. Root cause signature matching across 5 categories.
2. Semantic deduplication and incident clustering.
3. Broadcast notification advisory generation.
4. Latency SLA accounting and multi-model pricing formulas.
5. JSON and Markdown report export integrity.
"""

import os
import shutil
import tempfile
import unittest

from src.incident_triage import (
    IncidentClusterer,
    RootCauseCategory,
    IncidentSeverity,
    SupportIncident,
)
from evals.model_economics_benchmarker import (
    SupportEconomicsBenchmarker,
    SupportWorkloadResult,
)


class TestIncidentTriage(unittest.TestCase):
    """Test suite verifying automated incident clustering and root cause classification."""

    def setUp(self):
        self.clusterer = IncidentClusterer(cluster_threshold=0.25)

    def test_root_cause_classification_infrastructure(self):
        query = "502 Bad Gateway error when loading app dashboard, service unavailable."
        rc, sev, comps = self.clusterer.classify_root_cause(query)
        self.assertEqual(rc, RootCauseCategory.INFRASTRUCTURE_OUTAGE.value)
        self.assertEqual(sev, IncidentSeverity.P0_CRITICAL.value)
        self.assertIn("API Gateway", comps)

    def test_root_cause_classification_billing(self):
        query = "Stripe payment declined on checkout page, duplicate charge on card."
        rc, sev, comps = self.clusterer.classify_root_cause(query)
        self.assertEqual(rc, RootCauseCategory.BILLING_GATEWAY.value)
        self.assertEqual(sev, IncidentSeverity.P1_HIGH.value)
        self.assertIn("Stripe Gateway", comps)

    def test_root_cause_classification_auth(self):
        query = "Cannot login with corporate Okta SSO: SAML token expired 403."
        rc, sev, comps = self.clusterer.classify_root_cause(query)
        self.assertEqual(rc, RootCauseCategory.AUTH_CREDENTIAL.value)
        self.assertEqual(sev, IncidentSeverity.P1_HIGH.value)
        self.assertIn("Identity Provider", comps)

    def test_root_cause_classification_app_defect(self):
        query = "Uncaught TypeError null pointer exception when clicking PDF export button."
        rc, sev, comps = self.clusterer.classify_root_cause(query)
        self.assertEqual(rc, RootCauseCategory.APPLICATION_DEFECT.value)
        self.assertEqual(sev, IncidentSeverity.P2_MEDIUM.value)
        self.assertIn("Web Dashboard", comps)

    def test_root_cause_classification_user_config(self):
        query = "How do I configure IP whitelist firewall settings in dashboard?"
        rc, sev, comps = self.clusterer.classify_root_cause(query)
        self.assertEqual(rc, RootCauseCategory.USER_CONFIGURATION.value)
        self.assertEqual(sev, IncidentSeverity.P3_LOW.value)

    def test_incident_semantic_clustering(self):
        # First ticket creates incident
        res1 = self.clusterer.process_ticket("T-1", "Stripe payment declined on checkout page")
        self.assertFalse(res1["is_clustered"])
        inc_id = res1["incident_id"]

        # Second related ticket should merge into the same incident
        res2 = self.clusterer.process_ticket("T-2", "Payment failed during checkout with Stripe error")
        self.assertTrue(res2["is_clustered"])
        self.assertEqual(res2["incident_id"], inc_id)
        self.assertEqual(res2["cluster_ticket_count"], 2)

    def test_broadcast_advisory_generation(self):
        res = self.clusterer.process_ticket("T-10", "503 service unavailable server outage")
        self.assertIn("STATUS ADVISORY", res["broadcast_banner"])
        self.assertIn("investigating elevated errors", res["broadcast_banner"])

    def test_incident_summary_aggregation(self):
        self.clusterer.process_ticket("T-1", "502 Bad Gateway server crash")
        self.clusterer.process_ticket("T-2", "Stripe payment failed on checkout")
        self.clusterer.process_ticket("T-3", "Stripe payment declined error")

        summary = self.clusterer.get_incident_summary()
        self.assertEqual(summary["total_tickets_processed"], 3)
        self.assertEqual(summary["total_active_incidents"], 2)
        self.assertEqual(summary["total_clustered_tickets"], 3)


class TestSupportEconomicsBenchmark(unittest.TestCase):
    """Test suite verifying cross-model support token economics and deflection ROI."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmarker = SupportEconomicsBenchmarker(output_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workload_cost_computation_gemini(self):
        # 1,000,000 prompt ($0.075), 1,000,000 completion ($0.30) -> $0.375 total per 1M
        workload = SupportWorkloadResult(
            workload_id="test",
            workload_name="Test",
            iterations=1,
            prompt_tokens=1_000,
            completion_tokens=1_000,
        )
        cost_1k = workload.cost_per_1k_tickets("gemini_2_5_flash")
        # 1,000 runs of (1k prompt + 1k comp) = 1M prompt + 1M comp = $0.375
        self.assertAlmostEqual(cost_1k, 0.375, places=3)

    def test_workload_cost_computation_claude_and_gpt(self):
        workload = SupportWorkloadResult(
            workload_id="test",
            workload_name="Test",
            iterations=1,
            prompt_tokens=1_000,
            completion_tokens=1_000,
        )
        # Claude: 1k runs of (1k + 1k) = 1M prompt ($3.00) + 1M comp ($15.00) = $18.00
        cost_claude = workload.cost_per_1k_tickets("claude_3_5_sonnet")
        self.assertAlmostEqual(cost_claude, 18.00, places=2)

        # GPT-4o: 1M prompt ($2.50) + 1M comp ($10.00) = $12.50
        cost_gpt = workload.cost_per_1k_tickets("gpt_4o")
        self.assertAlmostEqual(cost_gpt, 12.50, places=2)

    def test_benchmark_engine_run_all_workloads(self):
        results = self.benchmarker.run_benchmarks(iterations=5)
        self.assertEqual(len(results), 4)

        for r in results:
            self.assertEqual(r.iterations, 5)
            self.assertTrue(r.latency_min_ms > 0)
            self.assertTrue(r.latency_p95_ms >= r.latency_min_ms)
            self.assertTrue(r.total_tokens > 0)

        summary = self.benchmarker.generate_summary(results)
        self.assertEqual(summary["workloads_evaluated"], 4)
        self.assertTrue(summary["fleet_average_p95_latency_ms"] > 0)
        self.assertTrue(summary["annual_500k_tickets_roi"]["annual_savings_vs_human_usd"] > 3_000_000)

    def test_export_json_and_markdown(self):
        results = self.benchmarker.run_benchmarks(iterations=3)
        summary = self.benchmarker.generate_summary(results)

        json_path = self.benchmarker.export_json(summary, "test_summary.json")
        self.assertTrue(json_path.exists())

        md_path = self.benchmarker.export_markdown(summary, "test_summary.md")
        self.assertTrue(md_path.exists())
        content = md_path.read_text(encoding="utf-8")
        self.assertIn("# Google L5 Executive Scorecard", content)
        self.assertIn("Comparative Enterprise Economics", content)
        self.assertIn("Google Gemini 2.5 Flash", content)

    def test_incident_clusterer_singleton(self):
        from src.incident_triage import get_incident_clusterer
        c1 = get_incident_clusterer()
        c2 = get_incident_clusterer()
        self.assertIs(c1, c2)

    def test_benchmarker_singleton(self):
        from evals.model_economics_benchmarker import get_economics_benchmarker
        b1 = get_economics_benchmarker()
        self.assertIsNotNone(b1)


if __name__ == "__main__":
    unittest.main()
