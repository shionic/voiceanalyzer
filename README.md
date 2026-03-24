# VoiceAnalyzer

VoiceAnalyzer is a Python project for **acoustic voice analysis**, **speaker embedding extraction**, and **database-backed voice matching**.

It includes:
- a low-level voice feature extractor (`voice_analyzer.py`)
- batch ingestion into PostgreSQL + pgvector (`process_batch.py`, `db.py`)
- speaker embedding utilities using SpeechBrain ECAPA (`ml_funcs.py`)
- a CLI matcher (`voice_match_cli.py`)
- a Telegram bot matcher (`voice_match_bot.py`)
- extra vocoder/re-synthesis experiments (`pyworld_revocoder.py`, `formants_revocoder.py`, `mfcc_revocoder.py`)

---

## Project Structure

- `voice_analyzer.py` — core acoustic analysis (pitch, MFCC, formants, spectral, energy)
- `db.py` — PostgreSQL access layer and schema initialization (with `pgvector`)
- `process_batch.py` — recursive or metadata-driven ingestion pipeline into DB
- `metadata_file.py` — metadata formats (`.json`, `.jsonl`, `.csv`) read/write/validation
- `process_metadata.py` — converts Mozilla Common Voice metadata to internal format
- `ml_funcs.py` — embedding extraction + cosine similarity utilities
- `speaker_pipeline.py` — shared pipeline used by CLI and Telegram bot
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
python voice_analyzer.py input.wav --format summary
```

Useful flags:
- `--include-frames` for frame-level detail
- `--sample-rate`, `--hop-length`, `--n-mfcc`, `--n-formants`, `--min-f0`, `--max-f0`

### B) Batch ingest directory to DB

```bash
python process_batch.py \
  --input /path/to/audio \
  --tags interview,english \
  --author "Speaker Name" \
  --db-host localhost --db-port 5432 \
  --db-name voiceanalyzer --db-user voiceanalyzer --db-password 1111
```

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

- `pyworld_revocoder.py` — extract/save/load WORLD features and resynthesize
- `formants_revocoder.py` — WORLD + Praat/Parselmouth formant emphasis
- `mfcc_revocoder.py` — MFCC -> reconstructed waveform via Griffin-Lim

Examples:

```bash
python pyworld_revocoder.py --input_wav in.wav --feature_dir feats --output_wav out.wav --mode both
python formants_revocoder.py in.wav out.wav
python mfcc_revocoder.py in.wav out.wav
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
