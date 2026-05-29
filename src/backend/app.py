from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_service import BertAIDetector, BertAIObfuscator, DetectorConfig, ObfuscatorConfig


DEFAULT_MODEL_PATH = "trained_models/ensemble_models_2_3000/manifest.json"


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    detector_type: Literal["ensemble", "hf"] | None = None
    model_path: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_length: int | None = Field(default=None, ge=16)
    batch_size: int | None = Field(default=None, ge=1)
    device: str | None = None


class ScoreResponse(BaseModel):
    results: list[dict]


class ObfuscateRequest(ScoreRequest):
    rewrite_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    window_size: int | None = Field(default=None, ge=1)
    stride: int | None = Field(default=None, ge=1)
    anchor: Literal["first", "center", "last"] | None = None
    aggregation: Literal["max", "mean"] | None = None


class ObfuscateResponse(BaseModel):
    results: list[dict]


def default_config() -> DetectorConfig:
    return DetectorConfig(
        detector_type=os.getenv("DETECTOR_TYPE", "ensemble"),
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        threshold=float(os.getenv("THRESHOLD", "0.5")),
        max_length=int(os.getenv("MAX_LENGTH", "256")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        device=os.getenv("DEVICE") or None,
    )


@lru_cache(maxsize=4)
def get_detector(config: DetectorConfig) -> BertAIDetector:
    return BertAIDetector(config)


def request_config(request: ScoreRequest) -> DetectorConfig:
    base = default_config()
    return DetectorConfig(
        detector_type=request.detector_type or base.detector_type,
        model_path=request.model_path or base.model_path,
        threshold=base.threshold if request.threshold is None else request.threshold,
        max_length=base.max_length if request.max_length is None else request.max_length,
        batch_size=base.batch_size if request.batch_size is None else request.batch_size,
        device=request.device if request.device is not None else base.device,
    )


def obfuscator_config(request: ObfuscateRequest) -> ObfuscatorConfig:
    return ObfuscatorConfig(
        rewrite_threshold=0.8 if request.rewrite_threshold is None else request.rewrite_threshold,
        window_size=4 if request.window_size is None else request.window_size,
        stride=1 if request.stride is None else request.stride,
        anchor=request.anchor or "center",
        aggregation=request.aggregation or "max",
    )


app = FastAPI(title="BERT AI Detector API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    config = default_config()
    return {"status": "ok", "default_config": asdict(config)}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        detector = get_detector(request_config(request))
        results = [asdict(result) for result in detector.score_batch(request.texts)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScoreResponse(results=results)


@app.post("/obfuscate", response_model=ObfuscateResponse)
def obfuscate(request: ObfuscateRequest) -> ObfuscateResponse:
    try:
        detector = get_detector(request_config(request))
        obfuscator = BertAIObfuscator(detector, obfuscator_config(request))
        results = [asdict(result) for result in obfuscator.obfuscate_batch(request.texts)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ObfuscateResponse(results=results)


@app.post("/reload")
def reload_models() -> dict:
    get_detector.cache_clear()
    return {"status": "model cache cleared"}
