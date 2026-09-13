"""An explicit side-call policy must stop fallback before another provider resolves."""
from types import SimpleNamespace

import httpx
import openai
import pytest


@pytest.mark.parametrize("status", [402, 429])
def test_disabled_provider_fallback_does_not_resolve_a_second_client(monkeypatch, tmp_path, status):
    from hermes_cli.config import get_config_path
    from agent import auxiliary_client as ac
    from agent.auxiliary_provider_fallback import provider_fallback

    get_config_path().write_text("auxiliary:\n  compression:\n    allow_provider_fallback: false\n")
    response = httpx.Response(status, request=httpx.Request("POST", "https://fixture.invalid"))
    error = openai.APIStatusError("quota exceeded", response=response, body={"error": "quota exceeded"})
    route = SimpleNamespace(task="compression", tag="", resolved_provider="openai-codex",
                            final_model="gpt-5.6-sol", base_info="", client=None, main_runtime=None)
    # The provider-resolution boundary must never be crossed, even for capacity errors.
    def unexpected_resolution(*args, **kwargs):
        pytest.fail("provider resolution reached despite disabled fallback")
    monkeypatch.setattr(ac, "resolve_provider_client", unexpected_resolution)
    # A configured alternate is deliberate: policy must take precedence over the chain.
    with get_config_path().open("a") as stream:
        stream.write("    fallback_chain:\n      - provider: anthropic\n        model: fixture\n")
    assert list(provider_fallback(error, route)) == []
    assert ac._try_configured_fallback_for_unavailable_client("compression", "openai-codex") == (None, None, "")


def test_provider_fallback_remains_enabled_without_operator_restriction():
    from agent.auxiliary_provider_fallback import provider_fallback_allowed
    assert provider_fallback_allowed("compression") is True
