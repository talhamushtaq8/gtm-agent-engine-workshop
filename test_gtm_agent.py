import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class RunAgentTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_scoring_request_receives_signed_in_rep_context(self):
        request = "Score LEAD-71001 against OFFER-10005."
        for rep in agent.REP_IDS:
            with self.subTest(user_id=rep["rep_id"]):
                with patch.object(agent.gtm_agent, "invoke") as invoke:
                    invoke.return_value = {"messages": [AIMessage(content="Scored")]}
                    result = agent.run_agent(request, user_id=rep["rep_id"])
                messages = invoke.call_args.args[0]["messages"]
                self.assertEqual(messages[0]["role"], "system")
                self.assertIn(rep["name"], messages[0]["content"])
                self.assertIn(rep["email"], messages[0]["content"])
                self.assertEqual(messages[1], {"role": "user", "content": request})
                self.assertEqual(result["reply"], "Scored")

    def test_missing_rep_still_invokes_agent_without_rep_context(self):
        with patch.object(agent.gtm_agent, "invoke") as invoke:
            invoke.return_value = {"messages": [AIMessage(content="Hello")]}
            result = agent.run_agent("Hello", user_id="unknown-rep")
        self.assertEqual(
            invoke.call_args.args[0]["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(result["reply"], "Hello")

    def test_email_context_matches_default_sender(self):
        rep = agent.REP_IDS[1]
        sent = {}

        def invoke(inputs, config):
            context = inputs["messages"][0]["content"]
            self.assertIn(rep["name"], context)
            self.assertIn(rep["email"], context)
            sent.update(agent.send_prospect_email.func(
                prospect={"name": "Prospect", "email": "prospect@example.com"},
                subject="Demo",
                body=f"Let's schedule a demo.\n\n{rep['name']}",
                runtime=SimpleNamespace(config=config),
            ))
            return {"messages": [AIMessage(content="Sent")]}

        with patch.object(agent.gtm_agent, "invoke", side_effect=invoke):
            result = agent.run_agent("Email the prospect", user_id=rep["rep_id"])
        self.assertEqual(result["reply"], "Sent")
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertTrue(sent["body"].endswith(rep["name"]))

    def test_explicit_rep_lookup_remains_available(self):
        rep = agent.REP_IDS[0]
        result = agent.get_current_rep.func(
            runtime=SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}}),
        )
        self.assertEqual(result, {"rep": rep, "found": True})


if __name__ == "__main__":
    unittest.main()
