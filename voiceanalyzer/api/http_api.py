from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from voiceanalyzer.batch import AudioFileProcessor
from voiceanalyzer.embeddings import cosine_similarity
from voiceanalyzer.storage import VoiceDatabase


def _safe_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    if out.get("x_vector") is not None:
        out["x_vector"] = [float(v) for v in out["x_vector"]]
    for k in ("created_at", "updated_at"):
        if out.get(k) is not None:
            out[k] = str(out[k])
    return out


class CompareRequest(BaseModel):
    left_id: int
    right_id: int


class PushAudioRequest(BaseModel):
    file_path: str
    skip_existing: bool = True
    include_frames: bool = False
    split_long_audio: bool = False
    author: Optional[str] = None
    author_source: Optional[str] = None
    tags: Optional[list[str]] = None
    reliable_quality_rating: Optional[float] = None
    unreliable_quality_rating: Optional[float] = None


class UpdateQualityRatingsRequest(BaseModel):
    id: int = Field(..., description="Record ID")
    reliable_quality_rating: Optional[float] = None
    unreliable_quality_rating: Optional[float] = None


class VoiceHTTPAPIServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        db_config: Optional[Dict[str, Any]] = None,
        internal_api_token: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.db_config = db_config or {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "voice_analysis"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
        }
        self.internal_api_token = internal_api_token or os.getenv("INTERNAL_API_TOKEN", "")
        self._app = self._create_app()
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="VoiceAnalyzer HTTP API", version="1.0.0")

        def get_db() -> VoiceDatabase:
            return VoiceDatabase(**self.db_config)

        def verify_internal_token(x_internal_token: Optional[str] = Header(default=None)) -> None:
            if not self.internal_api_token:
                return
            if x_internal_token != self.internal_api_token:
                raise HTTPException(status_code=401, detail="Unauthorized")

        @app.get("/public/record/{record_id}")
        def get_public_record(record_id: int, db: VoiceDatabase = Depends(get_db)) -> Dict[str, Any]:
            rec = db.get_recording_by_id(record_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="Record not found")
            return {"record": _safe_record(rec)}

        @app.post("/public/compare")
        def compare_records(payload: CompareRequest, db: VoiceDatabase = Depends(get_db)) -> Dict[str, Any]:
            left = db.get_recording_by_id(payload.left_id)
            right = db.get_recording_by_id(payload.right_id)
            if left is None or right is None:
                raise HTTPException(status_code=404, detail="One or both records not found")

            lv = left.get("x_vector")
            rv = right.get("x_vector")
            if lv is None or rv is None:
                raise HTTPException(status_code=400, detail="One or both records have no x_vector")

            sim = float(cosine_similarity(lv, rv))
            return {
                "left_id": payload.left_id,
                "right_id": payload.right_id,
                "similarity": sim,
                "cosine_distance": 1.0 - sim,
            }

        @app.post("/internal/push-audio")
        def push_audio(
            payload: PushAudioRequest,
            _: None = Depends(verify_internal_token),
        ) -> Dict[str, Any]:
            file_path = Path(payload.file_path)
            if not file_path.exists() or not file_path.is_file():
                raise HTTPException(status_code=400, detail="file_path does not exist or is not a file")

            processor = AudioFileProcessor(
                db_config=self.db_config,
                verbose=False,
                skip_existing=payload.skip_existing,
                include_frames=payload.include_frames,
                split_long_audio=payload.split_long_audio,
            )

            record_id = processor.process_file(
                filepath=file_path,
                author=payload.author,
                author_source=payload.author_source,
                tags=payload.tags,
                reliable_quality_rating=payload.reliable_quality_rating,
                unreliable_quality_rating=payload.unreliable_quality_rating,
            )

            if record_id is None:
                return {"record_id": None, "status": "skipped_or_not_inserted"}

            db = get_db()
            rec = db.get_recording_by_id(record_id)
            return {"record_id": record_id, "record": _safe_record(rec) if rec else None}

        @app.post("/internal/upload-audio")
        async def upload_audio(
            file: UploadFile = File(...),
            skip_existing: bool = Form(True),
            include_frames: bool = Form(False),
            split_long_audio: bool = Form(False),
            author: Optional[str] = Form(None),
            author_source: Optional[str] = Form(None),
            tags: Optional[str] = Form(None),
            reliable_quality_rating: Optional[float] = Form(None),
            unreliable_quality_rating: Optional[float] = Form(None),
            _: None = Depends(verify_internal_token),
        ) -> Dict[str, Any]:
            suffix = Path(file.filename or "upload.wav").suffix or ".wav"
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="voice_upload_")
            os.close(fd)

            try:
                with open(temp_path, "wb") as dst:
                    while True:
                        chunk = await file.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

                parsed_tags = [t.strip() for t in tags.split(",")] if tags else None
                if parsed_tags is not None:
                    parsed_tags = [t for t in parsed_tags if t]

                processor = AudioFileProcessor(
                    db_config=self.db_config,
                    verbose=False,
                    skip_existing=skip_existing,
                    include_frames=include_frames,
                    split_long_audio=split_long_audio,
                )

                record_id = processor.process_file(
                    filepath=Path(temp_path),
                    author=author,
                    author_source=author_source,
                    tags=parsed_tags,
                    reliable_quality_rating=reliable_quality_rating,
                    unreliable_quality_rating=unreliable_quality_rating,
                )

                if record_id is None:
                    return {"record_id": None, "status": "skipped_or_not_inserted"}

                db = get_db()
                rec = db.get_recording_by_id(record_id)
                return {"record_id": record_id, "record": _safe_record(rec) if rec else None}
            finally:
                await file.close()
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        @app.post("/internal/update-unreliable-quality")
        @app.post("/internal/update-quality-ratings")
        def update_quality_ratings(
            payload: UpdateQualityRatingsRequest,
            db: VoiceDatabase = Depends(get_db),
            _: None = Depends(verify_internal_token),
        ) -> Dict[str, Any]:
            if payload.reliable_quality_rating is None and payload.unreliable_quality_rating is None:
                raise HTTPException(status_code=400, detail="At least one rating must be provided")

            updated = db.update_quality_ratings(
                record_id=payload.id,
                reliable_quality_rating=payload.reliable_quality_rating,
                unreliable_quality_rating=payload.unreliable_quality_rating,
            )
            if not updated:
                raise HTTPException(status_code=404, detail="Record not found")

            rec = db.get_recording_by_id(payload.id)
            return {"ok": True, "record": _safe_record(rec) if rec else None}

        return app

    def start(self) -> None:
        if self._uvicorn_server is not None:
            return

        config = uvicorn.Config(app=self._app, host=self.host, port=self.port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._uvicorn_server is None:
            return

        self._uvicorn_server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._uvicorn_server = None
        self._thread = None
