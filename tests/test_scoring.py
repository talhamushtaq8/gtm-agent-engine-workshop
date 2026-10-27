import json
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    from gtm_agent import gtm_agent as agent


class ScoringTests(unittest.TestCase):
    def setUp(self):
        tracing = patch.dict(os.environ, LANGSMITH_TRACING="false")
        tracing.start()
        self.addCleanup(tracing.stop)
        profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)
        scoring = patch.object(agent, "_scoring_llm")
        self.llm = scoring.start()
        self.addCleanup(scoring.stop)
        self.result = agent.ProspectScore(
            score=72,
            justification="Revenue fits; some required technologies are missing.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=50, segment_fit=76,
            ),
        )
        self.llm.invoke.return_value = self.result

    def test_ids_load_records_and_preserve_scoring_payload_and_output(self):
        pairs = [
            ("LEAD-71001", "OFFER-10005"),
            ("LEAD-39002", "OFFER-10004"),
            ("LEAD-90001", "OFFER-10007"),
        ]
        for prospect_id, offering_id in pairs:
            with self.subTest(prospect_id=prospect_id):
                record = agent.data_service.get_prospect_record(prospect_id)
                expected_profile = {"prospect_id": prospect_id, **record}
                offering = agent.data_service.get_offering(offering_id)
                result = agent.score_prospect.invoke({
                    "prospect_id": prospect_id, "offering_id": offering_id,
                })
                self.assertEqual(result, self.result.model_dump())
                self.assertEqual(self.llm.invoke.call_args.args[0], [
                    {"role": "system", "content": agent.SCORING_PROMPT},
                    {"role": "user", "content": (
                        "Offering:\n" + json.dumps(offering, indent=2)
                        + "\n\nProspect profile:\n" + json.dumps(expected_profile, indent=2)
                    )},
                ])
                self.assertEqual(
                    agent.build_prospect_profile.invoke({"prospect_id": prospect_id}),
                    {"prospect_profile": expected_profile, "found": True},
                )

    def test_missing_or_invalid_offering(self):
        valid = agent.data_service.get_offering("OFFER-10005")
        invalid = [None, {}]
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            invalid.append({**valid, field: None})
        for offering in invalid:
            with self.subTest(offering=offering):
                with patch.object(agent.data_service, "get_offering", return_value=offering):
                    self.assertEqual(
                        agent.score_prospect.invoke({
                            "prospect_id": "LEAD-71001", "offering_id": "missing",
                        }),
                        {"score": None, "error": "Cannot score without a valid offering."},
                    )
        self.llm.invoke.assert_not_called()

    def test_missing_prospect(self):
        self.assertEqual(
            agent.score_prospect.invoke({
                "prospect_id": "missing", "offering_id": "OFFER-10005",
            }),
            {"score": None, "error": "Prospect not found."},
        )
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )
        self.llm.invoke.assert_not_called()

    def test_cached_profile_is_retained_but_scoring_uses_current_tech_stack(self):
        prospect_id = "LEAD-71001"
        profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        cached = profile["prospect_profile"]
        current_stack = ["Snowflake", "Kafka"]
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=current_stack):
            self.assertEqual(
                agent.build_prospect_profile.invoke({"prospect_id": prospect_id}), profile,
            )
            agent.score_prospect.invoke({
                "prospect_id": prospect_id, "offering_id": "OFFER-10005",
            })
        content = self.llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(scored_profile, {**cached, "tech_stack": current_stack})
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": prospect_id}), profile,
        )
        self.assertEqual(cached["tech_stack"], agent.data_service.fetch_tech_stack(prospect_id))

    def test_scoring_tool_schema_requires_only_ids(self):
        schema = agent.score_prospect.args_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"prospect_id", "offering_id"})
        self.assertEqual(set(schema["required"]), {"prospect_id", "offering_id"})
        for field in schema["properties"].values():
            self.assertEqual(field["type"], "string")


if __name__ == "__main__":
    unittest.main()
