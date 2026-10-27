import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import data_service, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def send_email(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Book a demo",
            body="Please book a demo.",
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}}),
        )

    def assert_blocked(self, prospect, email):
        with patch("gtm_agent.gtm_agent.uuid.uuid4", side_effect=AssertionError("Message constructed")):
            result = self.send_email(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": email,
        })

    def test_source_disqualification_blocks_even_when_supplied_flag_is_false(self):
        for prospect_id in ("LEAD-50001", "LEAD-50002", "LEAD-50003", "LEAD-50004", "LEAD-50005"):
            with self.subTest(prospect_id=prospect_id):
                self.assert_blocked(
                    {"prospect_id": prospect_id, "email": "stale@example.com", "disqualified": False},
                    data_service.get_prospect_record(prospect_id)["email"],
                )

    def test_supplied_disqualification_blocks_without_source_record(self):
        for prospect_id in (None, "UNKNOWN"):
            with self.subTest(prospect_id=prospect_id):
                self.assert_blocked(
                    {"prospect_id": prospect_id, "email": "prospect@example.com", "disqualified": True},
                    "prospect@example.com",
                )

    def test_supplied_disqualification_blocks_even_when_source_is_qualified(self):
        self.assert_blocked(
            {"prospect_id": "LEAD-12853", "disqualified": True},
            data_service.get_prospect_record("LEAD-12853")["email"],
        )

    def test_qualified_prospect_is_sent(self):
        prospect = {"prospect_id": "LEAD-12853", **data_service.get_prospect_record("LEAD-12853")}
        result = self.send_email(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["to_name"], prospect["name"])
        self.assertEqual(result["from"], data_service.get_rep("rep_amills")["email"])
        self.assertEqual(result["subject"], "Book a demo")
        self.assertEqual(result["body"], "Please book a demo.")

    def test_missing_email_retains_existing_failure(self):
        result = self.send_email({"prospect_id": "LEAD-12853"})
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })


if __name__ == "__main__":
    unittest.main()
