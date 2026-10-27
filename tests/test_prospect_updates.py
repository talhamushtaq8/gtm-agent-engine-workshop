import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))

    def test_update_persists_and_refreshes_cached_profile(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                data_service.PROSPECTS["LEAD-39002"]["tech_stack"] = ["AWS"]
                data_service._PROFILES.clear()
                if cached:
                    agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
                result = agent.update_prospect_info.invoke({
                    "prospect_id": "LEAD-39002", "technology": "Terraform",
                })
                self.assertTrue(result["updated"])
                self.assertTrue(result["found"])
                self.assertEqual(data_service.fetch_tech_stack("LEAD-39002"), ["AWS", "Terraform"])
                profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
                self.assertEqual(profile["prospect_profile"]["tech_stack"], ["AWS", "Terraform"])

    def test_duplicate_technology_is_not_reported_as_updated(self):
        before = list(data_service.fetch_tech_stack("LEAD-39002"))
        result = data_service.update_prospect_info("LEAD-39002", "AWS")
        self.assertEqual(result, {"updated": False, "found": True, "tech_stack": before})
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002"), before)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(data_service.update_prospect_info("unknown", "Terraform"),
                         {"updated": False, "found": False})

    def test_update_then_score_uses_saved_stack(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        offering = data_service.get_offering("OFFER-10004")

        def score_saved_stack(messages):
            payload = messages[1]["content"]
            saved = json.loads(payload.split("\n\nProspect profile:\n", 1)[1])
            missing = set(offering["required_tech_stack"]) - set(saved["tech_stack"])
            match = 100 * (1 - len(missing) / len(offering["required_tech_stack"]))
            return agent.ProspectScore(
                score=match, justification="Missing: " + (", ".join(sorted(missing)) or "none"),
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=match, segment_fit=100,
                ),
            )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_saved_stack
            before = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
            agent.update_prospect_info.invoke({"prospect_id": "LEAD-39002", "technology": "Terraform"})
            after = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(before["rubric_breakdown"]["tech_stack_match"], 75)
        self.assertIn("Terraform", before["justification"])
        self.assertEqual(after["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertNotIn("Terraform", after["justification"])
