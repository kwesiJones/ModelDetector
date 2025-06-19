import random
import httpx
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from model_predictor import ModelPredictor
from prompt_generator import PromptGenerator, ComplexityLevel, TopicDomain
from provider_config import get_provider_config
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("AIModelDetectionAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading prompt generator and model predictor")
    app.state.promptgen = PromptGenerator()
    app.state.modelpredictor = ModelPredictor('roberta_final')
    logger.info("Startup complete")
    yield

app = FastAPI(
    title="AI Model Detection Orchestrator",
    version="2.0",
    lifespan=lifespan
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class PromptGenRequest(BaseModel):
    domain: Optional[str] = Field(default=None)
    complexity: Optional[str] = Field(default=None)
    count: int = Field(default=5, ge=1)

class OfflineDetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

class BatchOfflineDetectRequest(BaseModel):
    texts: List[str] = Field(min_items=1)

class OnlineDetectRequest(BaseModel):
    provider: str
    api_key: str
    randomize: Optional[bool] = Field(default=True)
    prompt_domain: Optional[str] = Field(default=None)
    prompt_complexity: Optional[str] = Field(default=None)

class TopKRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    k: int = Field(default=3, ge=1)

@app.post("/generate_prompts")
async def generate_prompts(request: PromptGenRequest):
    logger.info(f"Prompt generation requested: domain={request.domain}, complexity={request.complexity}, count={request.count}")
    try:
        if request.domain and request.complexity:
            prompts = app.state.promptgen.generate_prompts(
                TopicDomain(request.domain), ComplexityLevel(request.complexity), request.count)
        elif request.domain:
            prompts = app.state.promptgen.generate_mixed_prompts(request.count, TopicDomain(request.domain))
        elif request.complexity:
            random_domain = random.choice(list(TopicDomain))
            prompts = app.state.promptgen.generate_prompts(
                random_domain, ComplexityLevel(request.complexity), request.count)
        else:
            prompts = app.state.promptgen.generate_diverse_prompts(request.count)
        logger.info(f"Generated {len(prompts)} prompts")
        return {"prompts": prompts}
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", summary="Predict model from a single text")
def predict(request: OfflineDetectRequest):
    logger.info("Offline single prediction requested")
    try:
        result = app.state.modelpredictor.predict(request.text)
        logger.info(f"Offline prediction complete: {result['most_likely_model']}")
        return result
    except Exception as e:
        logger.error(f"Offline prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", summary="Predict models from a batch of texts")
def predict_batch(request: BatchOfflineDetectRequest):
    logger.info(f"Offline batch prediction requested: batch size={len(request.texts)}")
    try:
        results = app.state.modelpredictor.predict_batch(request.texts)
        logger.info("Offline batch prediction complete")
        return {"results": results}
    except Exception as e:
        logger.error(f"Offline batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_topk", summary="Top-k most likely models for a single text")
def predict_topk(request: TopKRequest):
    logger.info(f"Top-k prediction requested: k={request.k}")
    try:
        results = app.state.modelpredictor.top_k_predictions(request.text, k=request.k)
        logger.info("Top-k prediction complete")
        return {"top_k": results}
    except Exception as e:
        logger.error(f"Top-k prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/classes", summary="Get class names")
def get_classes():
    logger.info("Class names requested")
    return {"classes": app.state.modelpredictor.get_class_names()}

@app.post("/detect_online")
async def detect_online(request: OnlineDetectRequest):
    logger.info(f"Online detection requested: provider={request.provider}, randomize={request.randomize}")
    try:
        if request.randomize:
            prompts = app.state.promptgen.generate_diverse_prompts(1)
        elif request.prompt_domain and request.prompt_complexity:
            prompts = app.state.promptgen.generate_prompts(
                TopicDomain(request.prompt_domain), ComplexityLevel(request.prompt_complexity), 1)
        elif request.prompt_domain:
            prompts = app.state.promptgen.generate_mixed_prompts(1, TopicDomain(request.prompt_domain))
        elif request.prompt_complexity:
            random_domain = random.choice(list(TopicDomain))
            prompts = app.state.promptgen.generate_prompts(
                random_domain, ComplexityLevel(request.prompt_complexity), 1)
        else:
            prompts = app.state.promptgen.generate_diverse_prompts(1)
        prompt = prompts[0]
        logger.info(f"Generated prompt for online detection: {prompt}")
    except Exception as e:
        logger.error(f"Prompt generation error (online): {e}")
        raise HTTPException(status_code=400, detail=f"Prompt generation error: {e}")

    try:
        cfg = get_provider_config(request.provider)
        if cfg["url"] is None:
            logger.warning(f"Provider '{request.provider}' is not callable via API")
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{request.provider}' is not currently callable via API. Use offline detection for this model class."
            )
        headers = cfg["headers"](request.api_key)
        payload = cfg["format_request"](prompt)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(cfg["url"], headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            ai_response = cfg["extract_response"](data)
            used_class = cfg["class_name"]
            used_family = cfg["family"]
        logger.info(f"Provider API call complete: class={used_class}, family={used_family}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI API error: {e}")
        raise HTTPException(status_code=502, detail=f"AI API error: {e}")

    try:
        detection = app.state.modelpredictor.predict(ai_response)
        logger.info(f"Offline detection of provider response complete: {detection['most_likely_model']}")
    except Exception as e:
        logger.error(f"Detection service error: {e}")
        raise HTTPException(status_code=502, detail=f"Detection service error: {e}")

    return {
        "prompt": prompt,
        "ai_response": ai_response,
        "provider_class": used_class,
        "provider_family": used_family,
        "detection": detection
    }

@app.get("/health", summary="Health check")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}