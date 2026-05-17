import unittest

from llm_compress.global_research import GlobalResearchAgent
from llm_compress.research import ResearchPlan


class GlobalResearchTests(unittest.TestCase):
    def test_initial_state_fallback_without_client(self):
        agent = GlobalResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=2)
        state = agent.initial_state(suite_context={"repos": []}, default_max_llm_bytes=1234)
        self.assertEqual(state.seed_plan.model, "openrouter/auto")
        self.assertEqual(state.seed_plan.max_llm_bytes, 1234)
        self.assertEqual(state.seed_plan.candidate_count, 2)
        self.assertTrue(state.lessons)

    def test_update_drops_lossless_overrides(self):
        agent = GlobalResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=3)
        previous = agent.initial_state(suite_context={"repos": []}, default_max_llm_bytes=20_000)
        history = [
            {
                "repo": {"name": "unit"},
                "success": True,
                "ratio": 0.5,
                "final_plan": {
                    **ResearchPlan().to_json(),
                    "lossless_overrides": ["./repo_specific.py"],
                },
            }
        ]
        state = agent.update_state(
            suite_context={"repos": []},
            history=history,
            previous_state=previous,
            default_max_llm_bytes=20_000,
        )
        self.assertEqual(state.seed_plan.lossless_overrides, [])
        self.assertTrue(any("unit" in lesson for lesson in state.lessons))

    def test_update_does_not_lower_max_llm_bytes(self):
        agent = GlobalResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=3)
        previous = agent.initial_state(suite_context={"repos": []}, default_max_llm_bytes=20_000)
        history = [
            {
                "repo": {"name": "unit"},
                "success": False,
                "final_plan": {
                    **ResearchPlan().to_json(),
                    "max_llm_bytes": 1_000,
                },
            }
        ]
        state = agent.update_state(
            suite_context={"repos": []},
            history=history,
            previous_state=previous,
            default_max_llm_bytes=20_000,
        )
        self.assertGreaterEqual(state.seed_plan.max_llm_bytes, 20_000)


if __name__ == "__main__":
    unittest.main()
