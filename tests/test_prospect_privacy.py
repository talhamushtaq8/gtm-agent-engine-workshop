import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langsmith import tracing_context

from gtm_agent import data_service
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS

with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def assert_safe(self, value, record):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for key, sensitive_value in record["billing_qualification"].items():
            self.assertNotIn(key, serialized)
            self.assertNotIn(sensitive_value, serialized)

    def test_contacts_allow_only_present_contact_fields(self):
        fields = {"name", "email", "disqualified", "annual_revenue", "enrichment_source"}
        for pid, record in PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                result = agent.get_prospect.invoke({"prospect_id": pid})
                self.assertTrue(result["found"])
                self.assertEqual(result["prospect"], {
                    "prospect_id": pid,
                    **{k: v for k, v in record.items() if k in fields},
                })
                self.assert_safe(result, record)
        with patch.object(data_service, "get_prospect_record", return_value={
            "name": "Contact", "future_private_field": "private",
        }):
            self.assertEqual(agent.get_prospect.invoke({"prospect_id": "minimal"}), {
                "prospect": {"prospect_id": "minimal", "name": "Contact"}, "found": True,
            })

    def test_new_profiles_and_scoring_pipeline(self):
        scoring_llm = Mock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 75}
        offering = next(iter(OFFERINGS.values()))
        with patch.object(agent, "_scoring_llm", scoring_llm):
            for pid, record in PROSPECTS.items():
                with self.subTest(prospect_id=pid):
                    result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                    self.assertTrue(result["found"])
                    profile = result["prospect_profile"]
                    self.assertEqual(profile, {
                        "prospect_id": pid,
                        **{k: v for k, v in record.items() if k != "billing_qualification"},
                    })
                    self.assert_safe(result, record)
                    self.assert_safe(data_service._PROFILES[pid], record)
                    arguments = {"prospect_profile": profile, "offering": offering}
                    self.assert_safe(arguments, record)
                    self.assertEqual(agent.score_prospect.invoke(arguments), {"score": 75})
                    self.assert_safe(scoring_llm.invoke.call_args.args[0], record)

    def test_legacy_cache_is_filtered_before_trace_and_tool_return(self):
        for pid, record in PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                data_service._PROFILES[pid] = {"prospect_id": pid, **record}
                runs = []
                with tracing_context(enabled="local"):
                    result = data_service.get_profile_from_db(
                        pid, langsmith_extra={"on_end": runs.append},
                    )
                self.assertEqual(len(runs), 1)
                self.assert_safe(runs[0].outputs, record)
                self.assert_safe(result, record)
                self.assertEqual(result["prospect_profile"]["tech_stack"], record["tech_stack"])
                tool_result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                self.assertEqual(tool_result, {**result, "found": True})
                self.assert_safe(tool_result, record)

    def test_scoring_filters_untrusted_profiles_without_mutating_them(self):
        scoring_llm = Mock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 75}
        offering = next(iter(OFFERINGS.values()))
        with patch.object(agent, "_scoring_llm", scoring_llm):
            for pid, record in PROSPECTS.items():
                with self.subTest(prospect_id=pid):
                    profile = {"prospect_id": pid, **record}
                    agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
                    messages = scoring_llm.invoke.call_args.args[0]
                    self.assert_safe(messages, record)
                    serialized_profile = messages[1]["content"].split("Prospect profile:\n", 1)[1]
                    self.assertEqual(json.loads(serialized_profile), {
                        k: v for k, v in profile.items() if k != "billing_qualification"
                    })
                    self.assertIn("billing_qualification", profile)

    def test_missing_prospect_preserves_not_found_results(self):
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
