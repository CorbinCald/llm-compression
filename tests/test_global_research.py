import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_compress.global_research import GlobalResearchAgent, GlobalResearchState, load_global_research_file
from llm_compress.research import ResearchPlan


class GlobalResearchTests(unittest.TestCase):
    def test_initial_state_fallback_without_client(self):
        agent = GlobalResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=2)
        state = agent.initial_state(suite_context={"repos": []}, default_max_llm_bytes=1234)
        self.assertEqual(state.seed_plan.model, "openrouter/auto")
        self.assertEqual(state.seed_plan.max_llm_bytes, 1234)
        self.assertEqual(state.seed_plan.candidate_count, 2)
        self.assertTrue(state.lessons)

    def test_resume_state_fallback_without_client_uses_persisted_state(self):
        agent = GlobalResearchAgent(None, allowed_models=["openrouter/auto"], max_candidates=3)
        previous = GlobalResearchState(
            reason="persisted",
            lessons=["persisted lesson"],
            seed_plan=ResearchPlan(format_variant="component_literal_heavy"),
        )
        state = agent.resume_state(
            suite_context={"repos": []},
            history=[{"repo": {"name": "old"}}],
            previous_state=previous,
            default_max_llm_bytes=20_000,
        )
        self.assertEqual(state.reason, "persisted")
        self.assertEqual(state.lessons, ["persisted lesson"])
        self.assertEqual(state.seed_plan.format_variant, "component_literal_heavy")

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
        self.assertFalse(any("unit" in lesson for lesson in state.lessons))
        self.assertTrue(any("raw verification" in lesson for lesson in state.lessons))

    def test_load_global_research_file_drops_repo_specific_lessons(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "benchmark-global-research.json"
            path.write_text(
                json.dumps(
                    {
                        "state": {
                            "reason": "prior run",
                            "lessons": [
                                "Go files must preserve package declarations.",
                                "click shell_completion.py needs string forward refs.",
                                "express lib/response.js must delete opts.maxAge.",
                            ],
                            "seed_plan": ResearchPlan(model="openrouter/auto").to_json(),
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_global_research_file(
                path,
                allowed_models=["openrouter/auto"],
                max_candidates=3,
                default_max_llm_bytes=20_000,
            )

        self.assertIsNotNone(loaded)
        state, _ = loaded
        self.assertEqual(state.lessons, ["Go files must preserve package declarations."])

    def test_load_global_research_file_restores_state_and_history(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "benchmark-global-research.json"
            path.write_text(
                json.dumps(
                    {
                        "label": "after-unit",
                        "state": {
                            "reason": "prior run",
                            "lessons": ["persisted lesson"],
                            "seed_plan": {
                                **ResearchPlan(model="openrouter/auto").to_json(),
                                "format_variant": "component_literal_heavy",
                                "lossless_overrides": ["./ignored.py"],
                                "max_llm_bytes": 1000,
                            },
                        },
                        "history": [{"repo": {"name": "old"}, "success": True}],
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_global_research_file(
                path,
                allowed_models=["openrouter/auto"],
                max_candidates=3,
                default_max_llm_bytes=20_000,
            )

        self.assertIsNotNone(loaded)
        state, history = loaded
        self.assertEqual(state.lessons, ["persisted lesson"])
        self.assertEqual(state.seed_plan.format_variant, "component_literal_heavy")
        self.assertEqual(state.seed_plan.lossless_overrides, [])
        self.assertEqual(state.seed_plan.max_llm_bytes, 20_000)
        self.assertEqual(history[0]["repo"]["name"], "old")

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
