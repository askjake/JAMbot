# Authorization Language Policy

The server-side execution gate and the `tool_execution_audit.v1` record are the
primary controls. This policy is the secondary control: it stops the assistant
from *describing* authorization in ways that are false or that manufacture
credential-shaped strings.

The enforced text lives in `app/agent/agents/prompts/chat_system_prompt.txt`
under `<tool_execution_authorization_language>` and is regression-tested by
`tests/test_authorization_language_policy_v1.py`.

## Rule 1 — never fabricate authorization material

The assistant must never invent, guess, echo, reconstruct, or display:

* an authorization token value;
* a token hash, digest, checksum, or fingerprint;
* a token prefix or suffix;
* any other token-derived string;
* placeholder material presented as if it were real.

It must not ask a user to paste a token into chat, and must not repeat one that
appears. Hashing is not a mitigation: a digest of a secret is still
secret-derived material, so no hashing path exists anywhere in this layer.

## Rule 2 — authorization is only ever server-reported booleans or status

Authorization state is described only through the four server-side booleans
(`operator_authorized`, `heavy_tools_authorized`, `persistence_authorized`,
`mutation_authorized`) or a server-provided enumerated status. The assistant
does not infer, assume, or predict authorization.

## Rule 3 — do not overstate activation or execution

The following states are distinct and must not be blurred:

| State | Meaning |
| --- | --- |
| `ACTIVATION_REQUESTED` | A request was recorded. Nothing was activated. |
| `ACTIVATION_PENDING` | Authorization is required and not yet granted. |
| `ACTIVATION_ELIGIBLE` | Requirements met; binding has not yet changed. |
| `EXECUTION_BLOCKED` | The gate refused the call. Nothing ran. |
| `EXECUTION_ALLOWED` | The gate permitted the call. |
| `EXECUTION_COMPLETED` | The server reported a finished execution. |

Words such as "submitted", "in flight", "processing", "executing", "applied",
and "complete" may only be used when a server-provided status value says so.

Requesting activation is not activation. Being eligible is not being bound.
Being allowed is not having completed.

## Rule 4 — assistant prose is not execution evidence

The assistant must state, when it matters, that its own text is not proof of
execution, and must never claim a tool "was called" when no tool result is
present. If a real tool result is absent, it says so and stops instead of
simulating, summarizing, or predicting one.

## Rule 5 — discretion

Security implementation detail, audit schema internals, and gate source
behaviour do not belong in ordinary user answers. Authorization state is
mentioned only when it actually affects the outcome.

## Facade semantics

`diship_backend_activate_tool_binding` records a *request*. Its response is the
only authoritative statement about activation state and now reports:

```json
{
  "activation_status": "ACTIVATION_REQUESTED_PENDING_AUTHORIZATION",
  "activation_performed": false,
  "execution_performed": false,
  "authorization_material_accepted": false,
  "write_performed": false
}
```

No management facade accepts an authorization argument. Server-controlled
authorization fields are stripped from every model-facing tool schema by
`prepare_model_facing_tool()`, so tokens are never advertised as model
arguments in the first place.
