# Pug's temporary provider restriction

Pug pins main, auxiliary and delegated work to Codex while the operator's
`execution.mode` is `codex-only`. Main/delegation fallback lists are empty.
Each auxiliary task also sets `allow_provider_fallback: false` and an empty
`fallback_chain`. The flag prevents unavailable-client and runtime capacity
recovery from discovering another provider. It defaults to true, preserving
normal Hermes behavior. Same-provider transient retry behavior is unchanged.

Runtime auxiliary provider fallback lives in
`agent/auxiliary_provider_fallback.py::provider_fallback`, called by the shared
sync/async recovery ladder. This does not change approval requirements or
grant authority to task comments.
