import unittest

from llm_compress.research import ResearchAgent, ResearchPlan


class ResearchTests(unittest.TestCase):
    def test_plan_validation_clamps_and_bans_lossless_opt_outs(self):
        plan = ResearchPlan.from_json(
            {
                "model": "bad-model",
                "format_variant": "bad",
                "chunking_strategy": "bad",
                "chunk_size_lines": 9999,
                "candidate_count": 99,
                "lossless_overrides": ["./ok.py", "../bad.py", "/bad.py"],
                "max_llm_bytes": -1,
                "temperature": 99,
            },
            default_model="openrouter/auto",
            allowed_models=["openrouter/auto"],
            max_candidates=3,
            default_max_llm_bytes=20_000,
        )
        self.assertEqual(plan.model, "openrouter/auto")
        self.assertEqual(plan.format_variant, "component_v1")
        self.assertEqual(plan.chunking_strategy, "file")
        self.assertEqual(plan.chunk_size_lines, 400)
        self.assertEqual(plan.candidate_count, 3)
        self.assertEqual(plan.lossless_overrides, [])
        self.assertEqual(plan.max_llm_bytes, 20_000)
        self.assertEqual(plan.temperature, 1.2)

    def test_research_agent_fallback_without_client(self):
        agent = ResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=2)
        plan = agent.initial_plan(context={"default_max_llm_bytes": 1234})
        self.assertEqual(plan.model, "openrouter/auto")
        self.assertEqual(plan.max_llm_bytes, 1234)
        self.assertEqual(plan.candidate_count, 2)

    def test_research_fallback_does_not_add_lossless_overrides(self):
        agent = ResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=3)
        previous = ResearchPlan(lossless_overrides=["./bad.py"], max_llm_bytes=1000)
        plan = agent.next_plan(
            context={"default_max_llm_bytes": 20_000},
            history=[{"plan": previous.to_json()}],
        )
        self.assertEqual(plan.lossless_overrides, [])
        self.assertGreaterEqual(plan.max_llm_bytes, 20_000)


if __name__ == "__main__":
    unittest.main()
