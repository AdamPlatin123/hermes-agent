"""Inherited compression remains compatible with third-party summary hooks."""

import pytest

from agent.context_compressor import ContextCompressor


SUMMARY = "The external engine preserved the deployment decisions."


class LegacySummaryEngine(ContextCompressor):
    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context=""):
        self.summary_calls.append((turns_to_summarize, focus_topic, memory_context))
        return SUMMARY


class ExplicitSummaryEngine(LegacySummaryEngine):
    def _generate_summary(
        self, turns_to_summarize, focus_topic=None, memory_context="", bypass_cooldown=None,
    ):
        self.received_bypass = bypass_cooldown
        return super()._generate_summary(turns_to_summarize, focus_topic, memory_context)


class KeywordSummaryEngine(LegacySummaryEngine):
    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", **kwargs):
        self.received_bypass = kwargs["bypass_cooldown"]
        return super()._generate_summary(turns_to_summarize, focus_topic, memory_context)


@pytest.mark.parametrize("engine_class", [LegacySummaryEngine, ExplicitSummaryEngine, KeywordSummaryEngine])
@pytest.mark.parametrize("bypass_cooldown", [False, True])
def test_compress_dispatches_supported_summary_hook_keywords(monkeypatch, engine_class, bypass_cooldown):
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *args, **kwargs: 100_000)
    engine = engine_class(
        model="test/model", protect_first_n=2, protect_last_n=2, quiet_mode=True,
    )
    engine.summary_calls = []
    # Keep a real compressible middle window without mocking the boundary selection.
    engine.tail_token_budget = 100
    messages = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Turn {i}: " + "Deployment decision and supporting context. " * 20,
        }
        for i in range(20)
    ]

    compressed = engine.compress(
        messages, current_tokens=90_000, focus_topic="deployment",
        memory_context="Keep the rollback plan.", bypass_cooldown=bypass_cooldown,
    )

    assert len(engine.summary_calls) == 1
    turns, focus, memory = engine.summary_calls[0]
    assert turns
    assert focus == "deployment"
    assert memory == "Keep the rollback plan."
    assert len(compressed) < len(messages)
    assert any(SUMMARY in message.get("content", "") for message in compressed)
    assert compressed[-1]["content"] == messages[-1]["content"]
    if engine_class is not LegacySummaryEngine:
        assert engine.received_bypass is bypass_cooldown
