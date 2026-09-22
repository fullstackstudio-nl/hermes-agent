# This fork

`fullstackstudio-nl/hermes-agent` is a fork of `NousResearch/hermes-agent` (MIT). It exists so FullStack Studio can run a gateway per customer with a few changes that upstream has not taken. The app and the plugin are built against plain upstream Hermes and must keep working there.

## How it is kept

- `main` is a copy of upstream. Never commit to it. Update it with `git fetch upstream && git push origin upstream/main:main`.
- `fss` carries our changes, rebased onto `main` whenever upstream moves. Rebase often; a small delta stays cheap.
- Every change is one commit with a message that says why, and a row in the table below. A change that upstream takes is dropped from the fork.
- Anything that can live in the Hermie plugin belongs in the plugin, not here.

## What we changed

| Commit | What | Why | Can it go? |
| --- | --- | --- | --- |
| `fix(tui_gateway): bind the signed-in user into the dashboard session context` | The gateway passes the session's authenticated user into `set_session_vars`, so `HERMES_SESSION_USER_ID`, `_ID_ALT` and `_USER_NAME` are filled instead of empty. | Without it a bot cannot say who it is talking to: the model reads those variables, finds them empty, and treats the person as unidentified. Every consumer benefits, not only our plugin. | Yes — it is offered upstream as pull request #118439. Drop this commit if that is merged. |
