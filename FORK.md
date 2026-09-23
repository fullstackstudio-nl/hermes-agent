# This fork

`fullstackstudio-nl/hermes-agent` is a fork of `NousResearch/hermes-agent` (MIT). It exists so FullStack Studio can run a gateway per customer with a few changes that upstream has not taken. The app and the plugin are built against plain upstream Hermes and must keep working there.

## How it is kept

- `main` carries our changes. `hermes update` installs the branch `main` when none is named, and the
  dashboard's update button never names one, so our code has to be what `main` points at — otherwise
  an update replaces a running deployment with plain upstream and takes the memory isolation below
  with it. Putting our code here makes the default correct for every host, a fresh clone included,
  without anyone having to know a setting.
- `upstream-main` is the copy of upstream and nothing else. Never commit to it. Update it with
  `git fetch upstream && git push origin upstream/main:upstream-main`.
- Bring upstream in by MERGING `upstream-main` into `main`. Never rebase `main` and never force-push
  it. Deployments run `main` now: a rebase rewrites the branch under them, and `hermes update` on a
  checkout that is on `main` and cannot fast-forward runs `git reset --hard origin/main` without
  tagging or stashing first. Merging keeps every update a fast-forward. Merge often; a small delta
  stays cheap.
- Nothing that runs this fork commits to `main`. A commit made on the branch an update targets is
  discarded by that reset, with no prompt and no recovery tag: the updater writes a rescue ref only
  when the two histories share no ancestor at all, and tags nothing otherwise. Naming the branch does
  not help — the parked-branch guard returns early when the checkout is already on the target, so
  `--branch <name>` while on `<name>` is the reset path too. Work that belongs to one deployment goes
  on a branch of its own, left checked out, with `updates.parked_branch_strategy: update_in_place` in
  that deployment's `config.yaml`. That is the one arrangement where the updater merges `origin/main`
  into the branch rather than resetting, and the only one that tags the commit it merges from.
- `fss` is held at the same commit as `main` while installs made from it move across. It is not a
  second line of development: it advances with `main` or not at all, and it is deleted once nothing
  is checked out on it.
- Never add a remote named `upstream` to a checkout that RUNS this fork. On the branch `main`,
  `hermes update --check` prefers a remote by exactly that name and compares the checkout against the
  real upstream, so it reports an update that is permanently available and must never be installed
  (`hermes_cli/update_cmd.py`). The name is matched literally, so a remote called anything else is
  invisible to the updater and is how you keep upstream fetchable. The apply path is already safe
  once `main` carries commits upstream does not have: it counts `upstream/main..origin/main` and
  skips the upstream sync while that is non-zero (`hermes_cli/update_cmd_git.py`).
- Every change is one commit with a message that says why, and a row in the table below. A change that upstream takes is dropped from the fork.
- Anything that can live in the Hermie plugin belongs in the plugin, not here.

## What we changed

Both memory and delivery fixes below were running as hand-applied patches on a live gateway (`~/.hermes/patches-hermes/`) and would have been lost at the next update. They belong here.

| Commit | What | Why | Can it go? |
| --- | --- | --- | --- |
| `fix(tui_gateway): bind the signed-in user into the dashboard session context` | The gateway passes the session's authenticated user into `set_session_vars`, so `HERMES_SESSION_USER_ID`, `_ID_ALT` and `_USER_NAME` are filled instead of empty. | Without it a bot cannot say who it is talking to: the model reads those variables, finds them empty, and treats the person as unidentified. Every consumer benefits, not only our plugin. | Yes — it is offered upstream as pull request #118439. Drop this commit if that is merged. |
| `feat(tui_gateway): name the signed-in user instead of printing their login id` | The WS ticket carries the provider's verified `display_name` beside the login it belongs to, and the gateway binds it as `HERMES_SESSION_USER_NAME`. A credential without a name keeps today's fallback, the bare login id. | The commit above only had the login id to bind, and on an OIDC deployment that is an opaque uuid — a bot could name the account but not the person. The name exists only on the request that mints the ticket: the turn holds no token, and calling the provider from a turn would put a network round trip on the hot path. | Only if upstream widens the minted WS identity to carry a name. Offer it upstream together with #118439; it builds on that commit, so the two are dropped together. |
| `test(tui_gateway): guard the session profile binding on the dashboard route` | A test that `_set_session_context` binds `HERMES_SESSION_PROFILE` from the record's own `profile_home`, and that the persistent-Docker container key follows it. No production change: this base already passes `profile=`. | It was unbound on our older base, and nothing fails loudly when it is: a cleared contextvar is authoritative, so readers silently fall back to the process's own profile and every dashboard session shares one sandbox. The API-server route has such a guard upstream; this route did not. | Yes, as soon as upstream has its own test for this route. Offer it upstream — it guards upstream's own behaviour, not ours. |
| `fix(tui_gateway): bind no user at all on a session two logins share` | `HERMES_SESSION_USER_ID`, `_ID_ALT` and `_USER_NAME` bind `""` on a session more than one signed-in person could be behind: a record is marked shared the moment a different login attaches and stays marked after that client leaves, and a transport slot carrying a login the record's stamp does not name counts the same way. One person in two windows, and a slot naming no login at all (parked, stdio, the compute-host child's own pipe), are unchanged. | The two commits above bind `auth_user_id`, the login the session RECORD was created under. Nothing re-stamps it when a second window attaches -- the attach path only logs the foreign login, and the slot becomes a `FanoutTransport`, which carries no `auth_identity` of its own -- so on a shared session the vars keep naming whoever opened the conversation. Everything downstream takes a bound name as verified: attribution (kanban, `send_message`, cron job args, the background-watcher fields) and per-person authorisation limits alike. Empty is a state all of them already handle; wrong is one none of them can question. | Only together with the commit below, which makes the shared case correct rather than merely empty. Offer both upstream with #118439: the hole is upstream's own, not fork-specific. |
| `fix(tui_gateway): attribute a turn to the connection that submitted it` | `prompt.submit` reads the submitting connection's own minted WS identity and carries it into the turn (a ContextVar bound beside the transport for the turn thread; a queued prompt carries it in its envelope, and two people's prompts no longer merge into one; the compute-host frame ships it to the child). `_acting_auth_user` resolves per connection first -- the turn's submitter, else the connection making this request (never for a dispatch the gateway made itself; see the row below) -- and only falls back to the record's stamp, unambiguous, when no signed-in connection is in play. A turn nobody submitted binds a sentinel so a watching peer is never mistaken for the person who asked. | The commit above makes the shared session honest but anonymous, and a gateway where two partners share one chat then has no attribution at all. The submitting connection is the one thing that does answer "who asked": `WSTransport.auth_identity` is minted at the upgrade from a verified ticket and no RPC param can reach it. The turn cannot read it itself -- it binds the SESSION's slot, a `FanoutTransport` naming nobody -- so it has to be read at submit and handed down, which is also what keeps it server-minted end to end. Needs a gateway restart and no configuration. | Together with the commit above, once upstream binds identity per turn rather than per session record. Offer both with #118439. |
| `fix(tui_gateway): never read a login off the socket that only relayed a turn` | A `prompt.submit` carrying any in-process param (the contract's underscore-aliased excluded fields -- `_turn_author`, `_hosted_task`, `_hosted_terminal_callback`) resolves to no submitter, so a relayed bot DM and a hosted-room turn name nobody. Every ContextVar a turn binds now sits inside the try whose finally resets it, transport included. The compute-host frame splits the two identities: `auth_user_id` is the conversation's own login (what the child builds the agent with) and `turn_auth_user_id` is who asked, resolved on the gateway because only it knows the session is shared. | The commit above read the identity off whatever socket carried the request. For a relay that is the socket of the person who relayed somebody else's message -- so a bot DM forwarded through one partner's desktop ran as that partner, two lines after the handler refused to let a client name an author at all. Keyed on the underscore prefix rather than a list of three names, so the next internal caller inherits it. The binding order matters for the same reason: `_acting_auth_user` reads the bound transport when no turn is in scope, so a transport leaked past a raising turn makes later work on that thread name that socket's owner. And without the frame split an isolated turn wrote memory in the submitter's scope while an inline turn on the same session wrote in the creator's -- two rules for one conversation. | With the two commits above; it is the same fix finished. |
| `Isolate team memory and preserve shared OAuth rotation` (carried, `4a01402f`) | mem0 recall is scoped per profile: one query per allowed `agent_id` instead of one wide query, and every row is re-checked against the scope it came back from. `mem0_update` and `mem0_delete` check the stored row's owner, the OSS history database moves into the profile's own home (0700), an embedding-dimension mismatch raises instead of deleting the collection, and `auth.json` is resolved before the lock and the atomic write. | The same leak our own smaller fix closed, plus the two halves it left open. Read access to a shared layer used to confer update and delete, because the tools trusted a `memory_id` the model had seen in a search result. Profiles also shared one mem0 history database, and a changed embedder silently dropped the whole collection. The auth half is what keeps one shared OAuth grant rotating: without resolving the link first, a refresh replaced the symlink with an independent stale copy and the lock no longer covered the profiles that share it. | Offer it upstream; none of it is fork-specific. The auth half stands alone and can go separately. |
| `Scope dashboard memory status and selection to profile` (carried, `850313fc`) | `GET /api/memory` and `PUT /api/memory/provider` take a `profile` and run inside that profile's scope instead of the launch profile's. | Without it the dashboard reports the launch profile's memory provider whatever profile is being looked at, and activating a provider writes to the wrong profile's config. | On this base only the regression test is left: upstream added the `profile` argument and `config_scoped_to_thread` itself, so the carried production change was already in place and we kept upstream's. Drop the test once upstream has its own for this route. |
| `Fix vulnerable dashboard dependencies` (carried, `75a3302c`) | `colord` and `sanitize-html` are pinned in `overrides` and raised in the lockfile (`2.9.3` to `2.10.0`, `2.17.6` to `2.17.7`). No package is added or removed and nothing else changes version. | Both were flagged against the dashboard's dependency tree. The `overrides` pins matter more than the lockfile bump: they stop a transitive dependency pulling a vulnerable version back in on the next resolve. | Yes, as soon as upstream's own lockfile carries both at or above these versions. |
| `fix(memory): keep the shared layer a setting on the carried isolation` | The default recall scope is the profile's own agent plus the layer named by `shared_agent_id` (`MEM0_SHARED_AGENT_ID`, or `shared_agent_id` in `mem0.json`, default `shared`). The name is validated like any other entry in the scope. | The carried isolation defaults the scope to the profile's own agent alone, so a shared layer only exists where a profile spells it out in `search_agent_ids`. A gateway with a profile per customer or per department still needs one layer everyone may read, and a deployment that already holds memories under such a layer would lose recall of them the moment the default narrowed — silently, because nothing is renamed or migrated in the store. | Only together with the carried isolation above; it has no meaning without it. |
| `feat(memory): tell the model which memories are its own and which are shared` | The prompt block and the four mem0 tool descriptions say that recall covers the profile's own memories plus shared memory it can read but not write, and that everything it stores lands in its own. Derived from the resolved scope: a profile with no shared memory is told nothing about any, and one whose own `agent_id` is the shared name is told that others read what it stores. No scope name is ever interpolated. | "You have memory" stopped being the whole truth once recall covered two scopes, and a model fills that gap wrongly in both directions -- presenting a fact recalled from the shared scope as something the user told it, and offering to save something for everyone, which no write path can do. A write always attaches the profile's own `agent_id`, so shared memory is read-only for an agent by construction, and the model has to be told so. | Only while we carry the isolation above; there is no second scope to describe without it. |
| `feat(memory): let a turn store into shared memory when it says so` | `mem0_add` takes a `shared` parameter where a shared layer is configured and readable: that one fact is stored under the layer's name with `written_by` in its metadata naming the writing profile. `mem0_update` and `mem0_delete` accept a shared entry from such a profile; every other scope stays owner-only. The automatic turn sync never shares. `MEM0_SHARED_WRITES=false` keeps a layer read-only. | A layer nobody can write is decoration: a write attaches the writing profile's own `agent_id`, so before this the only way to fill one was to run a profile whose own id was the layer's name. The parameter makes sharing a deliberate act for one fact rather than a property of the profile, which is what keeps a conversation from filling a layer other profiles read. | Offer it upstream with the isolation; it is the other half of the same feature. |
| `feat(tui_gateway): write the author of a turn onto the row it persists` | A user row written for a submitted turn carries `display_metadata.author` = `{"id": "<provider>:<user id>", "name"?}`, the identity of the connection that submitted it, and `gateway.capabilities` advertises `per_message_author` so a client knows whether this gateway attributes messages at all before it draws anything. Nothing is written for a turn nobody submitted, or for a transport that names no login. | A client cannot tell who wrote a message it did not send. `message.start` is contracted as an empty payload and two clients on one session share a `FanoutTransport`, so the socket says nothing about the other person's turn; the row has no author column and no wire field, so a reload leaves nothing to read either. Every client therefore treats an unmarked user row as its own, and a colleague's sentence is painted as the reader's -- a false statement, not a gap, and one no client can repair for itself. The commit above resolves who submitted a turn but keeps it in process; this writes that one fact down where it survives, in the free-form `display_metadata` dict every row already has and both the REST read and the WS history projection already forward whole. It fails closed the same way as the two commits above: an absent author is a state every reader handles, and nothing is written rather than something guessed. | Only if upstream gives a message an author of its own. The right shape is a `messages.author_id` column, and that is a migration a fork should not carry -- so the key is named `author` rather than anything fork-specific, and the whole change is upstreamable as it stands. Offer it with #118439 once the two identity commits above land. |
| `fix(bots): retry a delivery that failed because the target was busy` | `target_busy` becomes a constant, joins `ALL_REASONS` and is auto retryable. | A delivery that fails because the receiver is mid-turn is temporary. Without it such a message is dropped silently; raising `turn_wait_seconds` does not help, because deliveries serialise on the receiver's chat. | Yes, it is a plain bug fix. Offer it upstream. |
| `feat(profiles): cap how many profiles a gateway may hold` | A `profiles.max` setting in `config.yaml` (unset or `0` = unlimited, the existing behaviour). `create_profile()` — the one function `hermes profile create`, the dashboard's `POST /api/profiles`, and the `profiles.create` RPC all call — refuses once the default profile plus every live named one (exactly what `hermes profile list` shows) already reaches that count, with one message across all three doors: "This gateway allows N profiles and already has N." The check reads the count-then-create without a lock, so two simultaneous creates can both pass it and land one profile over the limit; that race is called out in the code rather than papered over with new locking. | Nothing in the fork or upstream stops the number of profiles on one gateway from growing without bound, and an operator who wants a hard ceiling had no setting to reach for. Guarding the single function every entry point already funnels through, instead of each caller separately, is what keeps a limit from silently having a second door nobody enforces. | Yes, it is a plain operator setting with nothing fork-specific in it. Offer it upstream as-is. |
| `feat(dashboard_auth): carry the signed-in user's email and picture from the ID token` | `/api/auth/me` keeps returning `email`, now `""` when the ID token says `email_verified: false` (boolean or the string `"false"`); the display-name fallback only uses a kept email. A `picture` claim is fetched at sign-in -- the browser callback and the desktop sign-in alike -- and never again while the session is used; the gateway stores its copy under its own home as `dashboard_auth/pictures/<sha256 of provider:sub>`. A sign-in that gets to fetch replaces it: a new image, or none when it brought no picture or its fetch failed. A sign-in never waits for a fetch slot: when the same person already has a fetch running, or all 4 slots are busy, it skips the fetch and the stored picture stays as it is. The last sign-in wins, not the last fetch to finish: a fetch stores or deletes only while its sign-in is still that person's latest, so an older stalled fetch can never overwrite or delete what a newer sign-in kept. A desktop sign-in whose pending authorization is gone fails before any of this and leaves the stored picture alone. `/api/auth/me` adds `picture_url` (`/api/auth/picture?id=<provider:sub>`, relative like every `/api` path) only while one is stored. That endpoint serves it to any signed-in user of this gateway and refuses anyone else (401); every id without a stored picture gets the same 404; the type sent is the one read from the bytes, with `nosniff` and `Content-Security-Policy: default-src 'none'; sandbox`; there is no way to upload one. The fetch takes `https` on port 443 only, refuses a URL carrying credentials, IDNA-encodes a non-ASCII host (an ASCII host, underscores included, passes as it is), and refuses the name if ANY resolved answer is private, loopback, link-local, shared (CGNAT), special-purpose, documentation, benchmarking, multicast or reserved space. Those are explicit deny-lists, not `is_global`, so the verdict does not change with the Python release. An IPv6 form carrying an IPv4 address (mapped, compatible, translated, 6to4, and NAT64 under the well-known `64:ff9b::/96` only -- an operator's own NAT64 prefix is not recognised) is judged by that IPv4 address; Teredo, site-local and everything outside global unicast is refused outright. It dials only vetted addresses, at most 8 of them, and checks every redirect again (at most 3; a redirect to plain `http` is refused). One deadline (10 s from the moment the sign-in hands the fetch over) bounds the lookup, every connect, the TLS handshake and every read and write, so a blackholed host or a peer trickling bytes -- over a real socket too -- cannot hold it longer; a lookup that hangs is abandoned on a daemon thread and cannot hold up a restart. The response must declare PNG, JPEG, WebP or GIF and its bytes must be one, with both dimensions at most 4096 read from the file header (unreadable is refused), and streaming stops past 512 KB. The fetch runs under a thread limiter of its own, never on the dashboard's shared threadpool tokens; the login waits for it at most the deadline plus 2 s and then goes ahead whatever the fetch thread is doing. None of it can fail a login. Email and picture come only from the ID token the gateway verified; values a client adds to the callback or to `/api/auth/me` are ignored. | The owner wants his own email and picture in the app. Hot-linking the provider's URL would have every device tell the provider whose chat it is looking at, and those URLs expire, so the gateway keeps a copy. **That is the trade-off: the gateway now stores a copy of each signed-in person's profile picture.** **Colleagues on a shared gateway see each other's name and picture.** That is deliberate: an author stamp already names the person, and the picture is served by that same id. `/api/auth/me` only ever describes the caller, so no colleague gets anyone else's email. Only the ID token is read, never userinfo: the session is rebuilt from that token on every request, which is what keeps the gate free of server-side session state, so a provider that puts `email` or `picture` only in userinfo gives the user neither. The fetch rules are this module's own rather than `tools/url_safety.py`, which honours `security.allow_private_urls` and the proxy environment for the agent's own browsing; a URL out of a token must be refused regardless. The slots, the one-fetch-per-person rule and the deadline exist because a person who can set their own picture URL at the provider could otherwise hang their own logins, queue everyone else's behind them, or -- about forty at once -- take the dashboard's shared threadpool. The cost is that on a busy gateway a picture change shows up one sign-in later. | Offer the email filter upstream on its own; it is a plain correctness fix. The picture store only if upstream wants avatars at all -- it adds state to a gate that has none today. |


## Identity: what is attributed to whom

Two questions with deliberately different answers, because conflating them is what the identity
commits above were fixing.

**Attribution follows the TURN.** `HERMES_SESSION_USER_ID`, `_ID_ALT` and `_USER_NAME` name the person
whose connection submitted the turn that is running — read at `prompt.submit` from the WS-upgrade
credential the server minted, carried into the turn, and never taken from anything a client sends. They
name nobody at all in three cases, and empty is a state every reader already handles: an ungated
gateway, a session more than one signed-in person could be behind, and a turn the gateway dispatched
itself (a relayed bot DM, a hosted-room turn, a crash continuation, a wake-up, a cron run) — the socket
such a request arrives on belongs to whoever relayed it, not to whoever asked.

**Memory follows the CONVERSATION.** The agent is built with the session record's own `auth_user_id`,
the login the conversation was opened under, so a chat two people share keeps its memories in one place
instead of splitting them per speaker. An isolated (compute-host) turn reaches both answers the same
way: the frame carries the record's stamp as `auth_user_id` for the agent and the submitter separately
as `turn_auth_user_id`.

Two consequences we accept rather than fix. Both are in the operator's own two-people-in-one-chat
scenario, so they are decisions, not oversights:

- **The record's stamp is re-minted when a conversation is resumed.** A live record is reused while
  anyone is attached, but once the last client drops, the record is parked and the next
  `session.resume` stamps a fresh one with whoever resumed. So the memory scope of a shared chat
  follows whoever reopened it last. Attribution is unaffected — it never reads the stamp for a turn a
  connection submitted. Fixing it means deciding what a shared conversation's memory scope IS, which
  is a product question and not this fix's to answer.
- **`display.busy_input_mode` (default `interrupt`) redirects a message typed mid-turn INTO the
  running turn.** B's sentence then executes inside A's turn, under A's identity, and no per-turn
  identity can change that because there is no second turn to attribute — the words became part of
  A's. This is the default and it bypasses the path above entirely. A gateway where two people share
  chats and attribution has to hold should set `display.busy_input_mode: queue` in `config.yaml`: a
  queued message runs as its own turn and carries its own sender. `steer` has the same problem as
  `interrupt`.

## Commits carried from another deployment

Three commits below were written on another deployment of this fork. They keep their original
subject and author date, and each one says in its own message that it was carried and which commit it
came from there. The author field is ours: the identity on those commits was a tool's, and no
assistant or tool name belongs anywhere in a repository we publish — so the credit is recorded in
words instead of in metadata. The table lists them by that subject and short sha, so a later rebase
can tell them from ours. Two of them needed a decision on the way in:

- Their memory isolation replaces the smaller fix we wrote for the same leak, so ours is not stacked
  on top of it. Our two earlier memory commits stay in history — `fix(memory): keep mem0 recall inside
  the profile's own agents` also introduces the `TARGET_BUSY` constant that the delivery fix depends
  on, so it must not be reverted — but their code, not ours, is what runs.
- Two behaviours of our own version are deliberately gone. `search_agent_ids: []` used to mean "recall
  every agent under this `user_id`" and now fails the profile closed instead, and a bare string is no
  longer accepted in place of a list. An empty or malformed scope is a configuration mistake, and the
  safe reading of a mistake is no recall rather than everyone's.
- Their tests assumed their own scope configuration and a default scope of one agent. The fixtures were
  adapted, and the two assertions that counted a recall as exactly one backend query now count one per
  allowed agent, which is what the carried recall does.

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
`search_agent_ids` explicitly ignores this setting — list the layer in that array instead. That list
must always contain the profile's own `agent_id`. An empty list, a missing `agent_id`, a padded name
or `*` is refused outright and leaves the profile no recall at all; it no longer means "everything".

Recall costs one backend query per name in the scope, so the default is two: the profile's own agent
and its shared layer.

### Shared writes

Writing is on wherever a layer is configured and readable, because configuring a layer is already the
opt-in. A deployment that wants a curated layer -- read by every profile, filled only by an operator
-- sets one line in the same place as the name:

```
MEM0_SHARED_WRITES=false
```

Two consequences worth knowing before turning a layer on. A profile that may write a layer may also
edit and delete anything in it, including an entry another profile wrote; `written_by` in an entry's
metadata is what makes that traceable, and entries written before this change do not carry it. And the
write target is the layer named by `shared_agent_id`, which must be one the profile can actually read
-- so a profile that sets `search_agent_ids` by hand must name the layer with `MEM0_SHARED_AGENT_ID`
or `shared_agent_id` as well, or it can read the layer and not write it.

The value is read when the agent is built, so restart the gateway (or open a new session) after
adding the line.
