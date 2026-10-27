import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class GTMAgentTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        })
        env.start()
        self.addCleanup(env.stop)
        self.agent = importlib.import_module("gtm_agent.gtm_agent")

    def run_request(self, responses, request):
        responses = iter(responses)
        calls = []

        def generate(messages, **kwargs):
            calls.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.object(ChatOpenAI, "_generate", side_effect=generate):
            result = self.agent.run_agent(
                request, user_id=self.agent.REP_IDS[0]["rep_id"],
                environment="test", thread_id="test-thread",
            )
        return result, calls

    def test_current_rep_without_harness_tools(self):
        rep = self.agent.REP_IDS[0]
        result, calls = self.run_request([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content=rep["email"]),
        ], "Who am I?")

        self.assertEqual(result["reply"], rep["email"])
        tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(json.loads(tool_results[0].content), {"rep": rep, "found": True})
        self.assertEqual(calls[0][0][0].content, self.agent.SYSTEM_PROMPT)
        self.assertEqual({tool["function"]["name"] for tool in calls[0][1]["tools"]}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        })
        rep_schema = next(t for t in calls[0][1]["tools"]
                          if t["function"]["name"] == "get_current_rep")
        self.assertNotIn("runtime", rep_schema["function"]["parameters"]["properties"])

    def test_prospect_scoring_flow(self):
        prospect_id = next(iter(self.agent.data_service.PROSPECTS))
        offering_id = next(iter(self.agent.data_service.OFFERINGS))
        profile = self.agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
        offering = self.agent.lookup_offering.invoke({"offering_id": offering_id})["offering"]
        score = self.agent.ProspectScore(
            score=75, justification="Fit assessed.",
            rubric_breakdown=self.agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        scoring_llm = Mock()
        scoring_llm.invoke.return_value = score
        with patch.object(self.agent, "_scoring_llm", scoring_llm):
            result, _ = self.run_request([
                AIMessage(content="", tool_calls=[{
                    "name": "get_current_rep", "args": {}, "id": "rep",
                }]),
                AIMessage(content="", tool_calls=[
                    {"name": "build_prospect_profile", "args": {"prospect_id": prospect_id}, "id": "profile"},
                    {"name": "lookup_offering", "args": {"offering_id": offering_id}, "id": "offering"},
                ]),
                AIMessage(content="", tool_calls=[{
                    "name": "score_prospect", "args": {"prospect_profile": profile, "offering": offering},
                    "id": "score",
                }]),
                AIMessage(content="Score: 75/100."),
            ], f"Score {prospect_id} for {offering_id}")

        self.assertEqual(result["reply"], "Score: 75/100.")
        tool_results = {m.name: json.loads(m.content) for m in result["messages"]
                        if isinstance(m, ToolMessage)}
        self.assertEqual(tool_results["get_current_rep"]["rep"], self.agent.REP_IDS[0])
        self.assertEqual(tool_results["build_prospect_profile"]["prospect_profile"], profile)
        self.assertEqual(tool_results["lookup_offering"]["offering"], offering)
        self.assertEqual(tool_results["score_prospect"], score.model_dump())
        scoring_input = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertIn(prospect_id, scoring_input)
        self.assertIn(offering_id, scoring_input)


if __name__ == "__main__":
    unittest.main()
