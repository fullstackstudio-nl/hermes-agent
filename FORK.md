# This fork

`fullstackstudio-nl/hermes-agent` is a fork of `NousResearch/hermes-agent` (MIT). It exists so FullStack Studio can run a gateway per customer with a few changes that upstream has not taken. The app and the plugin are built against plain upstream Hermes and must keep working there.

## How it is kept

- `main` is a copy of upstream. Never commit to it. Update it with `git fetch upstream && git push origin upstream/main:main`.
- `fss` carries our changes, rebased onto `main` whenever upstream moves. Rebase often; a small delta stays cheap.
- Every change is one commit with a message that says why, and a row in the table below. A change that upstream takes is dropped from the fork.
- Anything that can live in the Hermie plugin belongs in the plugin, not here.

## What we changed

Both memory and delivery fixes below were running as hand-applied patches on a live gateway (`~/.hermes/patches-hermes/`) and would have been lost at the next update. They belong here.

| Commit | What | Why | Can it go? |
| --- | --- | --- | --- |
| `fix(tui_gateway): bind the signed-in user into the dashboard session context` | The gateway passes the session's authenticated user into `set_session_vars`, so `HERMES_SESSION_USER_ID`, `_ID_ALT` and `_USER_NAME` are filled instead of empty. | Without it a bot cannot say who it is talking to: the model reads those variables, finds them empty, and treats the person as unidentified. Every consumer benefits, not only our plugin. | Yes — it is offered upstream as pull request #118439. Drop this commit if that is merged. |
| `feat(tui_gateway): name the signed-in user instead of printing their login id` | The WS ticket carries the provider's verified `display_name` beside the login it belongs to, and the gateway binds it as `HERMES_SESSION_USER_NAME`. A credential without a name keeps today's fallback, the bare login id. | The commit above only had the login id to bind, and on an OIDC deployment that is an opaque uuid — a bot could name the account but not the person. The name exists only on the request that mints the ticket: the turn holds no token, and calling the provider from a turn would put a network round trip on the hot path. | Only if upstream widens the minted WS identity to carry a name. Offer it upstream together with #118439; it builds on that commit, so the two are dropped together. |
| `fix(memory): keep mem0 recall inside the profile's own agents` | mem0 recall is filtered on `agent_id` as well as `user_id`, with a shared layer and a `search_agent_ids` setting. | Without it every profile under one principal recalls every other profile's memories. A gateway with a profile per department, or per customer, leaks between them. | Offer it upstream; the leak is not ours. |
| `refactor(memory): make the shared memory layer's name a setting` | The shared layer defaults to `shared` and is named by `MEM0_SHARED_AGENT_ID` (a profile's own `.env`) or `shared_agent_id` in that profile's `mem0.json`. | The former default was a company name in code other people run, and one fixed name cannot serve a host that wants one house layer, one layer per customer and one per department. Read through the profile secret scope like `MEM0_AGENT_ID`, so under multiplexing a profile's layer name comes from its own `.env`, never the launch profile's. | Only with the commit above; the two are one feature. |
| `fix(bots): retry a delivery that failed because the target was busy` | `target_busy` becomes a constant, joins `ALL_REASONS` and is auto retryable. | A delivery that fails because the receiver is mid-turn is temporary. Without it such a message is dropped silently; raising `turn_wait_seconds` does not help, because deliveries serialise on the receiver's chat. | Yes, it is a plain bug fix. Offer it upstream. |

## Deploying the shared-memory-layer rename

A gateway that already has memories under a differently named shared layer must name that layer, or
those memories stop being recalled: recall matches the layer by name, and nothing is renamed or
migrated in the store. Nothing is lost — the memories stay in mem0 and come back the moment the name
is set — but recall is silently narrower until it is, which is the case a customer of the hosted
product will hit after an update.

Per profile that should keep seeing an existing layer, add one line to that profile's own `.env`
(`~/.hermes/profiles/<profile>/.env`, or `~/.hermes/.env` for the default profile):

```
MEM0_SHARED_AGENT_ID=<the layer's existing name>
```

Put it in the profile's own `.env`, not in the gateway service's environment. A profile's secret
scope is built from that profile's own files and `MEM0_SHARED_AGENT_ID` is not on the global-env
allowlist in `agent/secret_scope.py`, so once the host multiplexes only the launch profile still
resolves the service environment and every other profile would read nothing. `"shared_agent_id"` in
that profile's `mem0.json` does the same and wins over the `.env` value. A profile that sets
`search_agent_ids` explicitly ignores this setting — list the layer in that array instead.

The value is read when the agent is built, so restart the gateway (or open a new session) after
adding the line.
