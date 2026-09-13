"""Provider fallback rung, shared by synchronous and asynchronous auxiliary calls."""
from __future__ import annotations
from agent import auxiliary_client as aux


def provider_fallback(first_err: Exception, route: aux._LadderRoute):
    """Last rung: other providers (per-task chain; then auto: main fallback chain + discovery
    chain, explicit: main-agent-model net). Returns the response or None.
    Capacity errors (payment/quota, connection, exhausted 429, model incompatible, malformed
    response) bypass the explicit-provider gate — the provider cannot serve this request
    regardless of user intent. Auth errors only fall back in auto mode."""
    task, tag, resolved_provider = route.task, route.tag, route.resolved_provider
    # Respect explicit provider choice for transient errors (auth, request validation, etc.) but allow
    # fallback when the provider clearly cannot serve the request due to capacity: payment/quota exhaustion
    # and connection failures are capacity problems, not request constraints. See #26803: daily token quota
    # (429 + "too many tokens per day") must fall back just like a 402 credit error.
    # Rate limits are included: after retries are exhausted, a 429 means the provider is at capacity. See
    # #52228. See #26803: daily token quota must fall back like a 402 credit error.
    is_auto = resolved_provider in {"auto", "", None}
    reason = next((label for predicate, label in aux._FALLBACK_REASONS if predicate(first_err)), None)
    is_capacity_error = any(
        predicate(first_err) for predicate, label in aux._FALLBACK_REASONS if label != "auth error")
    if reason is None or not (is_auto or is_capacity_error):
        return None
    if reason == "payment error":
        # Mark the concrete backend (not the "auto" label) unhealthy so later aux calls skip
        # it instead of paying another doomed RTT.
        aux._mark_provider_unhealthy(
            aux._recoverable_pool_provider(resolved_provider, route.client, main_runtime=route.main_runtime)
            or resolved_provider, base_url=route.base_info)
    aux.logger.info("Auxiliary %s%s: %s on %s (%s), trying fallback",
                task or "call", tag, reason, resolved_provider, first_err)
    # Skip only the failed model for model-specific failures; 401/402 are provider-wide, so
    # auth keeps skipping the credential surface, while billing is scoped to the endpoint:
    # separate custom URLs can carry separate credentials (or no billing relationship at all).
    _chain_failed_model = None if reason in ("auth error", "payment error") else route.final_model
    from agent.backend_identity import FailureScope
    _chain_failure_scope = (
        FailureScope.ENDPOINT
        if reason == "payment error" and aux._custom_health_base_url(resolved_provider, route.base_info)
        else None
    )
    fb_client, fb_model, fb_label = aux._try_configured_fallback_chain(
        task, resolved_provider or "auto", reason=reason, failed_model=_chain_failed_model,
        failed_base_url=route.base_info, failure_scope=_chain_failure_scope)
    if fb_client is None and is_auto:
        fb_client, fb_model, fb_label = aux._try_main_fallback_chain(
            task, resolved_provider or "auto", reason=reason, failed_model=_chain_failed_model,
            failed_base_url=route.base_info, failure_scope=_chain_failure_scope)
        if fb_client is None:
            fb_client, fb_model, fb_label = aux._try_payment_fallback(
                resolved_provider, task, reason=reason, failed_base_url=route.base_info,
                failure_scope=_chain_failure_scope)
    elif fb_client is None:
        fb_client, fb_model, fb_label = aux._try_main_agent_model_fallback(
            resolved_provider, task, reason=reason, failed_model=_chain_failed_model,
            failed_base_url=route.base_info, failure_scope=_chain_failure_scope)
    if fb_client is not None:
        # Second pass: the candidate credential was stale and quarantined — walk the discovery
        # chain once more (unhealthy entries are skipped).
        for _pass in range(2):
            aux._record_route_info(route.route_info, aux._fallback_provider_from_label(fb_label), fb_model)
            fb_resp = yield aux._LadderStep("fallback", (fb_client, fb_model, fb_label))
            if fb_resp is not None:
                return fb_resp
            if _pass == 0:
                fb_client, fb_model, fb_label = aux._try_payment_fallback(
                    resolved_provider, task, reason="stale fallback credential",
                    failed_base_url=route.base_info, failure_scope=_chain_failure_scope)
                if fb_client is None:
                    break
    # All fallback layers exhausted — one user-visible warning, then re-raise.
    aux.logger.warning("Auxiliary %s%s: %s on %s and all fallbacks exhausted "
                   # All fallback layers exhausted — emit a single user-visible warning so the operator
                   # knows aux task is about to fail. (#26882) The error itself is re-raised below.
                   # (#26882)
                   "(fallback_chain + main agent model). Raising original error.",
                   task or "call", tag, reason, resolved_provider)
    return None
