# VoiceAnalyzer

VoiceAnalyzer is a Python project for **acoustic voice analysis**, **speaker embedding extraction**, and **database-backed voice matching**.

It includes:
- a low-level voice feature extractor (`voiceanalyzer/analysis/voice_analyzer.py`)
- batch ingestion into PostgreSQL + pgvector (`process_batch.py`, `voiceanalyzer/storage/db.py`)
- speaker embedding utilities using SpeechBrain ECAPA (`voiceanalyzer/embeddings/ml_funcs.py`)
- a CLI matcher (`voice_match_cli.py`)
- a Telegram bot matcher (`voice_match_bot.py`)
- extra vocoder/re-synthesis experiments (`voiceanalyzer/revocoders/*.py`)

---

## Project Structure

- `voiceanalyzer/analysis/voice_analyzer.py` — core acoustic analysis (pitch, MFCC, formants, spectral, energy)
- `voiceanalyzer/storage/db.py` — PostgreSQL access layer and schema initialization (with `pgvector`)
- `process_batch.py` — recursive or metadata-driven ingestion pipeline into DB
- `voiceanalyzer/metadata/metadata_file.py` — metadata formats (`.json`, `.jsonl`, `.csv`) read/write/validation
- `process_metadata.py` — converts Mozilla Common Voice metadata to internal format
- `voiceanalyzer/embeddings/ml_funcs.py` — embedding extraction + cosine similarity utilities
- `voiceanalyzer/matching/speaker_pipeline.py` — shared pipeline used by CLI and Telegram bot
- `voiceanalyzer/audio/io.py` — shared audio loading helpers
- `voiceanalyzer/api/http_api.py` — FastAPI-based HTTP API (public + internal endpoints)
- `voiceanalyzer/batch/batch_processor.py` — batch processor internals
- `voice_match_cli.py` — local command-line matching utility
- `voice_match_bot.py` — Telegram bot endpoint
- `docker-compose.yml` / `Dockerfile` — containerized app + PostgreSQL (pgvector image)
- `example_metadata.json`, `example_metadata.csv` — metadata examples for batch processing

---

## Features

### 1) Voice analysis
Extracts:
- pitch statistics (mean/std/min/max/percentiles/voicing rate)
- energy statistics
- MFCC means
- spectral characteristics (centroid, bandwidth, flatness, rolloff, ZCR, RMS)
- LPC-based formant estimates (F1–F4)

### 2) Speaker embeddings
- Uses `speechbrain/spkrec-ecapa-voxceleb`
- Produces L2-normalized embedding vectors
- Computes cosine similarity for speaker comparison

### 3) Database storage and search
- Stores full analysis JSON (`JSONB`)
- Stores tags and metadata (author/source/quality ratings)
- Stores embeddings in `vector(192)` and supports similarity queries
- Provides filtering by author/tags and statistics

### 4) Matching workflow
- Analyze input audio
- Compare its embedding against DB records tagged `male` and `female`
- Return best matches plus feature deltas

---

## Requirements

- Python 3.11+ recommended
- PostgreSQL with `pgvector` extension
- FFmpeg/libsndfile available (for audio decoding)

Python dependencies are listed in `requirements.txt`.

---

## Quick Start (Docker)

1. Create optional `.env` values (or rely on defaults from `docker-compose.yml`):

```env
DB_NAME=voiceanalyzer
DB_USER=voiceanalyzer
DB_PASSWORD=1111
DB_PORT=5432
TELEGRAM_BOT_TOKEN=
```

2. Start services:

```bash
docker compose up --build
```

This starts:
- `db` (PostgreSQL + pgvector)
- `app` (default command runs `voice_match_bot.py`)

---

## Quick Start (Local Python)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Initialize database schema:

```bash
python process_batch.py --init-db \
  --db-host localhost --db-port 5432 \
  --db-name voiceanalyzer --db-user voiceanalyzer --db-password 1111
```

> Note: some modules default to `voice_analysis`/`postgres`; for consistency, pass DB options explicitly or set environment variables (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

---

## Usage

### A) Analyze a single file

```bash
python voiceanalyzer/analysis/voice_analyzer.py input.wav --format summary
```

Useful flags:
- `--include-frames` for frame-level detail
- `--sample-rate`, `--hop-length`, `--n-mfcc`, `--n-formants`, `--min-f0`, `--max-f0`

### B) Batch ingest directory to DB

```bash
python process_batch.py \
  --input /path/to/audio \
  --split-long-audio \
  --tags interview,english \
  --author "Speaker Name" \
  --db-host localhost --db-port 5432 \
  --db-name voiceanalyzer --db-user voiceanalyzer --db-password 1111
```

> `--split-long-audio` is optional. By default, files are processed as single trimmed clips.

### C) Batch ingest from metadata file

```bash
python process_batch.py \
  --input-meta example_metadata.json \
  --db-host localhost --db-port 5432 \
  --db-name voiceanalyzer --db-user voiceanalyzer --db-password 1111
```

Supported metadata formats:
- `.json`
- `.jsonl`
- `.csv`

Generate a template metadata file:

```bash
python process_batch.py --create-template template.json --template-format json
```

### D) Run speaker matching from CLI

```bash
python voice_match_cli.py /path/to/query.wav
python voice_match_cli.py /path/to/query.wav --json
```

### E) Run Telegram bot

```bash
export TELEGRAM_BOT_TOKEN="<your_token>"
python voice_match_bot.py
```

Send a voice message or audio file to the bot and it returns analysis + male/female proximity matches.

When bot starts, it also starts internal HTTP API server in the same process.

---

## HTTP API (started with bot)

The API now runs on **FastAPI + Uvicorn** under the hood, which makes routes easier to extend and maintain (request validation, typed payloads, cleaner error responses).

By default:
- host: `127.0.0.1`
- port: `8080`

Config env vars:
- `INTERNAL_API_HOST`
- `INTERNAL_API_PORT`
- `INTERNAL_API_TOKEN` (optional token via `X-Internal-Token` header for internal endpoints)

### Public endpoints

- `GET /public/record/{id}` — get DB record by id
- `POST /public/compare` — compare two records by id

Request body for compare:

```json
{ "left_id": 1, "right_id": 2 }
```

### Internal endpoints

- `POST /internal/push-audio` — push new audio into DB using same ingestion flow as batch processor (`AudioFileProcessor.process_file`)
- `POST /internal/upload-audio` — upload audio via `multipart/form-data` and ingest directly
- `POST /internal/update-unreliable-quality` — update `unreliable_quality_rating` by id

Request body examples:

```json
{ "file_path": "/abs/path/to/audio.wav", "author": "speaker", "tags": ["telegram"] }
```

```json
{ "id": 123, "unreliable_quality_rating": 0.71 }
```

Upload example:

```bash
curl -X POST "http://127.0.0.1:8080/internal/upload-audio" \
  -H "X-Internal-Token: <token_if_configured>" \
  -F "file=@/abs/path/to/audio.wav" \
  -F "author=speaker" \
  -F "tags=telegram,male"
```

---

## Metadata conversion helper

Convert Mozilla Common Voice `train.tsv` into internal metadata format:

```bash
python process_metadata.py mozilla_common_voice /path/to/commonvoice /path/to/output.json --format json
```

Output entries include:
- `filepath` (relative under `clips/`)
- `author` (`client_id`)
- `author_source` (`MozillaCommonVoices`)
- normalized age/gender tags
- computed unreliable rating from up/down votes

---

## Vocoder / Re-synthesis Utilities

These scripts are experimental and independent from the DB pipeline:

- `voiceanalyzer/revocoders/pyworld_revocoder.py` — extract/save/load WORLD features and resynthesize
- `voiceanalyzer/revocoders/formants_revocoder.py` — WORLD + Praat/Parselmouth formant emphasis
- `voiceanalyzer/revocoders/mfcc_revocoder.py` — MFCC -> reconstructed waveform via Griffin-Lim

Examples:

```bash
python voiceanalyzer/revocoders/pyworld_revocoder.py --input_wav in.wav --feature_dir feats --output_wav out.wav --mode both
python voiceanalyzer/revocoders/formants_revocoder.py in.wav out.wav
python voiceanalyzer/revocoders/mfcc_revocoder.py in.wav out.wav
```

---

## Notes and Caveats

- First run may be slow due to downloading SpeechBrain model weights.
- `process_batch.py` prompts interactively on metadata validation errors.
- `setup.sh` is interactive and intended as a guided local setup helper.
- Audio support depends on backend codecs (FFmpeg/libsndfile).

---

## License

MIT — see [LICENSE](LICENSE).
