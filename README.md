# Document Analyser

Local document upload and grounded chat using Streamlit, FastAPI, a separate
Python ingestion worker, PostgreSQL, Docling, GLiNER, Ollama and Qdrant.
Python 3.12 and `uv` are required.

## Start locally

```sh
uv sync
# Start PostgreSQL and Qdrant; skip the Qdrant service if you already run it.
docker compose up -d postgres qdrant
# Optional: copy .env.example to .env and edit it; do not overwrite an existing .env.
# Applications automatically load .env when run from the project root.
uv run alembic upgrade head
ollama pull nomic-embed-text:v1.5
ollama pull qwen3:8b
```

Run each application process in a separate terminal from the project root:

```sh
uv run uvicorn document_analyser.api.app:app --host 127.0.0.1 --port 8000
uv run python -m document_analyser.ingestion.worker
uv run streamlit run src/document_analyser/ui/app.py
```

Upload in the sidebar. The API preserves original bytes and queues work; the
worker must be running for progress to advance. The processing panel polls every
three seconds and shows the stage, approximate stage progress, errors and NER
warnings. Use **Refresh chat document list** when indexing finishes. Ask questions
or request “summarise this document” in chat. There is no separate summary action.
The first Docling PDF or GLiNER run may download models into their normal local
caches. Run one worker initially on the M1 Max to limit concurrent model memory.

## Application configuration

`src/document_analyser/config.py` defines typed settings shared by the API,
worker, Streamlit, migrations and CLI tools. See `.env.example` for every setting
and its default value. Configuration precedence is explicit constructor/function
overrides (where supported), environment variables, `.env`, then Python defaults.
Run from the project root: `.env` and relative storage paths resolve from the
working directory. Unknown dotenv keys are ignored. Use uppercase names for
consistent environment behaviour.

Settings are cached once per process. **Restart the API, worker and Streamlit**
after configuration changes. Alembic and CLI commands read settings each time
they start. Credentials are masked in the settings representation; they are never
included in processing-version snapshots. Keep `.env` out of Git.

| Group | Environment variables |
| --- | --- |
| Infrastructure | `DATABASE_URL`, `DOCUMENT_STORAGE_DIR`, `DOCUMENT_API_URL`, `QDRANT_URL`, `MAX_UPLOAD_BYTES`, `QDRANT_TIMEOUT` |
| Ollama embeddings | `OLLAMA_HOST`, `EMBEDDING_MODEL`, `EMBEDDING_TIMEOUT`, `EMBEDDING_BATCH_SIZE` |
| Generation | `GENERATION_MODEL`, `GENERATION_TIMEOUT`, `GENERATION_TEMPERATURE`, `GENERATION_CONTEXT_TOKENS`, `GENERATION_MAX_TOKENS` |
| Retrieval and summaries | `RETRIEVAL_LIMIT`, `SUMMARY_BATCH_CHARACTERS`, `SUMMARY_MAX_LEVELS` |
| Chunking | `CHUNK_MAX_WORDS`, `CHUNK_OVERLAP_WORDS`, `CHUNK_MAX_CHARACTERS` |
| NER | `NER_MODEL`, `NER_TOKENIZER`, `NER_LABELS`, `NER_THRESHOLD`, `NER_WINDOW_TOKENS`, `NER_OVERLAP_TOKENS` |
| Polling and UI requests | `WORKER_POLL_SECONDS`, `UI_POLL_SECONDS`, `API_REQUEST_TIMEOUT`, `UPLOAD_REQUEST_TIMEOUT`, `CHAT_REQUEST_TIMEOUT` |

Timeouts/poll intervals are positive seconds; limits/budgets are positive integers.
Chunk and NER window overlaps must be nonnegative and smaller than their respective
limits. Generation output tokens must be smaller than the context size. These
checks do not estimate prompt token counts: tune context and summary budgets for
the selected model. An evidence item larger than the summary budget raises a clear
error rather than being silently omitted. Temperature ranges from 0 to 2; NER
threshold ranges from 0 to 1. `NER_LABELS` is a JSON list, for example:

```dotenv
NER_LABELS='["person name","organization name","geographical location"]'
```

New versions snapshot chunking and NER settings at upload/retry time. Workers use
those saved settings even if the worker's environment subsequently changes. Older
versions missing NER fields use the original defaults. Retry an original to apply
new processing settings. Generation/retrieval settings apply on process restart;
changing the embedding model requires rebuilding the affected version's Qdrant
projection. The embedding adapter still uses Nomic prefixes: changing the model
name alone does not adapt the prefix strategy for unrelated embedding families.
Change the NER tokenizer along with its model when the encoder requires it.

API and worker must use the same database and storage root. Compose binds services
to localhost and uses persistent named volumes. Compose environment/port settings
are separate from Python settings; changing `DATABASE_URL` or `QDRANT_URL` changes
the application connection, not the container configuration. The example database
credentials are for local development. Back up PostgreSQL and storage together.

## Data ownership and failure handling

PostgreSQL owns document identity, original checksum and size, immutable processing
versions, jobs, ordered passages with Docling references and provenance, chunks
with passage-relative character offsets, NER candidates, and indexing status.
Alembic owns schema changes; application startup does not create tables.

Original files and Docling JSON artifacts are accessed through the `Storage`
protocol. `LocalStorage` uses immutable keys and materializes objects into worker
temporary directories. An object-store implementation can replace this adapter.
A failed database registration removes the just-written original; a hard process
crash between object creation and database commit can leave an unreferenced object
that requires later cleanup. Existing objects are never overwritten.

Workers claim queued jobs using PostgreSQL `FOR UPDATE SKIP LOCKED`, commit the
claim, and perform model work outside that transaction. Parsed passages and chunks
are committed before NER and indexing. Processing errors retain these records and
show the failed stage. Retry creates a fresh version from the preserved original;
old versions and artifacts remain available in the database/storage. The UI shows
the latest version for each original.

NER mentions are **unreviewed candidates** with label, score, offsets, model,
library version, tokenizer, labels, threshold and window configuration. They do
not influence retrieval or grounded answers. NER exceptions are rolled back and
recorded as warnings while embedding/indexing continues.

Qdrant is a rebuildable projection. Indexing uses idempotent point IDs, waits for
acknowledged writes, and marks a version indexed only after all chunks succeed.
Queries are scoped to the processing version and embedding identity. Chat resolves
retrieved IDs against PostgreSQL for source text and citations. Whole-document
summaries visit every ordered chunk in bounded batches, then consolidate section
findings. If consolidation does not shrink, section findings are returned to
preserve coverage. All chunks are considered; model-generated summaries can still
omit details.

Rebuild a version after losing Qdrant data, or after changing embedding configuration:

```sh
uv run python -m document_analyser.ingestion.worker --reindex VERSION_ID
```

After an interrupted worker, **stop all workers**, mark running jobs failed, then
retry them in the UI:

```sh
uv run python -m document_analyser.ingestion.worker --fail-interrupted
```

There is no automatic lease expiry or crash retry. Stage progress is coarse, and
long parsing, NER or indexing stages may remain at the same percentage. Partial
vector writes are never exposed as a ready version. A rebuild must run while no
other process is indexing the same version.

Existing JSON-based parsing/chunking/indexing and summary CLI modules remain
available. The web catalog now comes from PostgreSQL; upload the original files
to bring previously generated local JSON documents into the managed workflow.
Embedding and generation protocols remain provider independent.

## Checks

```sh
uv run pytest -q
uv run ruff check src/document_analyser/api src/document_analyser/persistence \
  src/document_analyser/ingestion/storage.py src/document_analyser/ingestion/worker.py \
  src/document_analyser/config.py src/document_analyser/extraction \
  src/document_analyser/generation/answer.py src/document_analyser/generation/chat.py \
  src/document_analyser/generation/ollama_provider.py src/document_analyser/generation/summarise.py \
  src/document_analyser/retrieval/chunking.py src/document_analyser/retrieval/search.py \
  src/document_analyser/retrieval/index_document.py src/document_analyser/retrieval/providers \
  src/document_analyser/retrieval/check_vector_search.py src/document_analyser/ui/app.py tests migrations
uv run alembic check
```

Tests use synthetic content, SQLite with foreign keys and local in-memory Qdrant.
They cover API registration, original preservation, failure/retry, nonfatal NER,
source offsets, projection rebuilding, real Docling Markdown/table conversion,
summary coverage, migration roundtrips, settings precedence/validation, provider
overrides, queued processing snapshots and configurable batching/retrieval limits. PostgreSQL row-lock concurrency and
live Ollama/GLiNER/PDF model execution require separate integration validation.
PostgreSQL migration revision `5950555a0300` was subsequently applied and verified
by the user; automated migration roundtrips use an isolated SQLite database. Existing port 6333 was occupied, so tests did not
modify the existing Qdrant server.

Documents and outputs under `data/`, environment secrets, virtual environments and
model caches are excluded from Git. Do not place personal documents or model
weights elsewhere in the repository.

Implementation references: [SQLAlchemy locking](https://docs.sqlalchemy.org/en/20/core/selectable.html#sqlalchemy.sql.expression.Select.with_for_update)
and [Alembic migrations](https://alembic.sqlalchemy.org/en/latest/tutorial.html).
