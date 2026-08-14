# ThoughtLoom API

The AI half of ThoughtLoom. FastAPI on Render, Cohere for the model, Supabase
for everything it reads and writes.

```
Flutter ──auth + CRUD──> Supabase (Row Level Security)
   │
   └──AI──> this service ──> Cohere
                │
                └──> Supabase (service-role key: reads context, writes turns)
```

## Running

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # or .venv/bin/pip on unix
cp .env.example .env                              # then fill it in
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Docs at `/docs`, health at `/health`.

```bash
.venv/Scripts/python -m pytest
```

The tests use a fake model and a fake Supabase — no network, no Cohere key, no
project. What they cover is the logic *around* the model: the round cap, the
option cleaning, the search decision, what gets written and when. The model's
own output is not something a test can assert.

## Deploying

Render, from this repo's `main`. No `render.yaml` — the service is configured in
the dashboard:

| | |
| --- | --- |
| Build | `pip install -r requirements.txt` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

`.python-version` pins 3.12.7 — the version the tests run on. Without it Render
picks its own default, which moves, and `supabase` and `langchain-cohere` are the
kind of dependencies that notice.

Environment: `COHERE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. Set
them in Render's dashboard, never in a committed file — see `.env.example` for
what each one is and why the service-role key must not be the anon one.

`/health` answers on a box with no configuration at all, so a service that has
not been given its keys yet still boots far enough to say so.

## Endpoints

| Endpoint | Does |
| --- | --- |
| `POST /api/adaptive-question` | Records the last answer, returns the next generated question — or `done` |
| `POST /api/recommendation` | Researches if needed, takes a position, persists it |
| `POST /api/follow-up` | One more turn of the conversation |
| `POST /api/complete-chat` | The user left: close it, name it, remember it |

Every endpoint requires `Authorization: Bearer <supabase access token>`.

The app's own reads and writes — the chat list, the search, the transcript — go
straight to Supabase under RLS and are not here. Nothing reaches this service
unless it needs the model.

### The conversation loop

`POST /api/adaptive-question` is both halves of the loop, so a step is one round
trip rather than two that can half-succeed:

```jsonc
// first call of a chat
{"chat_id": "..."}
// → {"done": false, "round": 1, "message_id": "...", "question": "...", "options": [...]}

// every call after
{"chat_id": "...", "answer": {"message_id": "...", "text": "The money"}}
// → the next question, or {"done": true, ...}
```

The question row is written *unanswered* when the model generates it, and its id
comes back to the client, which returns it with the answer. One row per turn, and
an abandoned chat still shows what it was in the middle of asking.

Calling again while a question is unanswered returns *that* question rather than
generating another — so the client's retry is safe to hit repeatedly.

Options are generated for the specific user and never include an "other": the
client always renders its own free-text fallback, and the fallback being
guaranteed is the app's promise rather than the model's.

The model stops when it has enough, usually after 3-6 questions.
`MAX_ADAPTIVE_ROUNDS` (8) is a hard ceiling it does not get to argue with.

### Closing a chat, and what carries to the next one

`POST /api/complete-chat {"chat_id": "..."}` returns as soon as the status is
written. Naming the chat and folding it into `user_memory` are two more model
calls, and they run in a **background task after the response** — the user has
just pressed Back, and once the request is here the work happens whether or not
their app survives the next second.

It is idempotent: titling is skipped for a chat that has a title, the merge for
one already in memory. The client calls it on leaving, and again if it later
opens a completed chat that never got named.

What it writes then reaches every later chat, through `app/services/context.py`:

| | |
| --- | --- |
| `user_memory` | Durable facts, merged not appended — the model is given what it already knew and returns the whole updated memory, so a fact can be *corrected* rather than contradicted. Split global / per-category. |
| Past chats | `app/services/recall.py` — same category, or the same words, scored and cut to three. No embeddings. |

**A first-time user gets neither block at all** — headers included, not
"(nothing yet)". A standing instruction to reference what you remember, handed
to a model with nothing to remember, invents a shared history. Cold start is the
one behaviour here worth guarding, and `tests/test_context.py` guards it.

## Security

**This service holds the Supabase service-role key, which bypasses Row Level
Security.**

In the Flutter client, RLS *is* the authorisation model — the policies in
`schema.sql` are the only thing stopping one user reading another's rows. This
service reads a chat's context and writes the model's turns back, and it is not
acting as any single user when it does. So RLS cannot help here, and something
else has to do its job.

That is `app/core/auth.py`: every request carries the caller's Supabase access
token, it is verified with Supabase, and the chat's `user_id` must match. Without
it, `POST /api/adaptive-question {"chat_id": "<anyone's>"}` would hand a
stranger's situation to whoever guessed a uuid.

**`user_memory` is the one thing chat ownership does not authorise.** Every other
row here is reached through a chat; memory is scoped to a *person*. Getting it
wrong would not leak one conversation — it would graft a stranger's life onto
someone's permanent record, and every future chat would be answered out of it.
So `merge_from_chat` takes the verified caller's id as a required argument and
refuses to run if the chat is not theirs. That check is redundant — `auth.py`
just proved it — and it stays: redundant here means two independent reasons this
cannot go wrong, and the first one being wrong once is unrecoverable.

Missing and forbidden chats both return 404 — that an id exists is not a fact an
attacker should get.

The service-role key must never reach a client build, a public repo, or a
committed file. Set it in Render's environment. If it leaks, rotate it.

## Swapping the model

Everything goes through `LanguageModel` in `app/core/llm.py` — `complete` and
`complete_json`. Add a class, change `get_model()`. No call site moves and no
prompt changes.

`ChatCohere` is the original setup this service started with, moved behind the
interface rather than replaced.

## Web search

The recommendation call decides for itself whether a question turns on current
real-world facts. Most do not — what someone owes their family does not live on
the internet — and searching those wastes a round trip and drags in noise. When
it does search, it writes its own queries, and the snippets go into the prompt.

DuckDuckGo via `ddgs`: no API key, no billing account. It is also the least
reliable link in the stack — no SLA, and it rate-limits — so everything in
`app/core/web_search.py` fails soft. A search that returns nothing produces
advice grounded only in what the user said, which is the product working slightly
worse rather than failing.

To swap providers, implement `WebSearch` and change `get_search()`.

## Layout

```
app/
  core/
    llm.py              the LanguageModel interface, and Cohere behind it
    auth.py             proves the caller owns the chat
    supabase_client.py  service-role reads and writes
    web_search.py       the WebSearch interface, and DuckDuckGo behind it
    config.py           environment
  services/
    context.py               profile + memory + past chats + transcript, for all prompts
    adaptive_engine.py       what to ask next, and when to stop
    recommendation_engine.py research → answer → persist; and follow-ups
    titling.py               naming a finished chat
    memory.py                folding a finished chat into what we know
    recall.py                which past chats connect to this one
  prompts/            the actual instructions. These carry the product
  api/                the routes
  schemas/            request/response models
tests/
```

`app/services/context.py` is the only place that turns rows into prompt text, so
the questions, the recommendation, and the follow-ups all reason over the same
picture of a person. Three prompts assembling it separately would drift, and the
bug — advice grounded in a subset of what was said — would be invisible.

`load_context(chat, recall=False)` skips the cross-chat reads. Titling and the
memory merge reason about one conversation and must not be handed the others.
