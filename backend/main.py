"""HTTP API for parsing datasets, running both real models, and evaluating them."""
from __future__ import annotations

import io
import json
import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from backend.model_service import ALLOWED, service

app = FastAPI(title="Digital Declutter AI API", version="1.0.0")
origins = os.getenv("DECLUTTER_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173,null").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


def metrics(actual: list[str], predicted: list[str]) -> dict:
    labels = ["KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"]
    p, r, f, _ = precision_recall_fscore_support(actual, predicted, labels=labels, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(actual, predicted), "precision": p, "recall": r, "f1": f,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(), "labels": labels}


def validate_items(items: list[dict]) -> list[dict]:
    cleaned = []
    for idx, row in enumerate(items):
        text = row.get("item_text") or row.get("text")
        if text is None or not str(text).strip():
            raise HTTPException(422, f"Row {idx + 1} is missing item_text")
        actual = row.get("actual_action")
        if actual is not None:
            actual = str(actual).strip().upper()
            if actual not in ALLOWED:
                raise HTTPException(422, f"Row {idx + 1} has unsupported actual_action '{actual}'")
        cleaned.append({**row, "item_text": str(text), "domain": row.get("domain", "unknown"), "actual_action": actual})
    return cleaned


@app.get("/api/health")
def health():
    return {"status": "ready", "provider": service.provider, "base_model": os.getenv("DECLUTTER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
            "fine_tuned_model": os.getenv("DECLUTTER_FINE_TUNED_MODEL", ""), "adapter_available": service.configured,
            "configuration_message": service.configuration_message}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    rows = validate_items(request.items)
    if not service.configured:
        raise HTTPException(503, service.configuration_message)
    try:
        base = service.predict_batch(rows, fine_tuned=False)
        service.unload()
        tuned = service.predict_batch(rows, fine_tuned=True)
        service.unload()
    except FileNotFoundError as exc:
        service.unload()
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        service.unload()
        raise HTTPException(503, f"Model inference failed: {exc}") from exc
    predictions = []
    for row, bp, tp in zip(rows, base, tuned):
        predictions.append({"item_text": row["item_text"], "domain": row["domain"], "actual_action": row.get("actual_action"), "base": bp, "fine_tuned": tp,
                            "correct": bool(row.get("actual_action") and tp["action"] == row["actual_action"]) if row.get("actual_action") else None})
    actual = [r["actual_action"] for r in rows]
    provenance = [bool(row.get("synthetic", False)) for row in rows]
    data_provenance = "synthetic" if provenance and all(provenance) else "mixed" if any(provenance) else "user_supplied"
    result = {"records": len(rows), "predictions": predictions, "metrics": None, "source": "model_inference", "dataset_provenance": data_provenance}
    if all(actual):
        result["metrics"] = {"base": metrics(actual, [p["base"]["action"] for p in predictions]), "fine_tuned": metrics(actual, [p["fine_tuned"]["action"] for p in predictions])}
    return result


@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    try:
        raw = await file.read()
        if name.endswith(".csv"):
            frame = pd.read_csv(io.BytesIO(raw))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            frame = pd.read_excel(io.BytesIO(raw))
        else:
            raise HTTPException(415, "Upload a CSV or Excel file (.xlsx/.xls).")
        if frame.empty:
            raise HTTPException(422, "The uploaded dataset has no records.")
        if "item_text" not in frame.columns:
            raise HTTPException(422, "The dataset must contain an item_text column with the item description for the model.")
        request = AnalyzeRequest(items=frame.where(pd.notnull(frame), None).to_dict(orient="records"))
        return analyze(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read dataset: {exc}") from exc


@app.post("/api/inspect-file")
async def inspect_file(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    try:
        raw = await file.read()
        if name.endswith(".csv"):
            frame = pd.read_csv(io.BytesIO(raw))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            frame = pd.read_excel(io.BytesIO(raw))
        else:
            raise HTTPException(415, "Upload a CSV or Excel file (.xlsx/.xls).")
        if "item_text" not in frame.columns:
            raise HTTPException(422, "Required column missing: item_text")
        actual = frame["actual_action"].dropna().astype(str).str.upper() if "actual_action" in frame.columns else []
        domain = "Digital Items"
        if "domain" in frame.columns and frame["domain"].nunique(dropna=True) == 1:
            domain = str(frame["domain"].dropna().iloc[0]).title()
        return {"records": int(len(frame)), "columns": [str(c) for c in frame.columns],
                "classes": int(actual.nunique()) if len(actual) else None,
                "has_labels": bool(len(actual) == len(frame) and len(actual) > 0), "domain": domain}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not inspect dataset: {exc}") from exc
