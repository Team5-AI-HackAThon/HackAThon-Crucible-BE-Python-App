from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

load_dotenv()

from app.routes import sentiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Video Intelligence API",
    description="Video sentiment analysis and pitch scoring using Azure Video Indexer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)


def _health_payload() -> dict:
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root():
    return _health_payload()


@app.get("/health")
async def health():
    """Liveness check for UIs, load balancers, and monitors."""
    return _health_payload()

@app.on_event("startup")
async def startup_event():
    print("🚀 Video Intelligence API starting up...")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 Video Intelligence API shutting down...")
