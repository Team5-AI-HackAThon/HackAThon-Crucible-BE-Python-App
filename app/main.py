from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

# Load environment variables early (before importing modules that read env vars)
load_dotenv()

from app.routes import chat, sentiment
from app.models import HealthResponse
from app.services.llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="LLM Chat API",
    description="POC for Multi-Provider LLM Chat API (OpenAI + Anthropic) with Performance Tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(sentiment.router)

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    llm_service = LLMService()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        providers_available=llm_service.get_available_providers()
    )

@app.on_event("startup")
async def startup_event():
    llm_service = LLMService()
    print("🚀 LLM Chat API starting up...")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Default Provider: {llm_service.default_provider}")
    print(f"Available Providers: {llm_service.get_available_providers()}")

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 LLM Chat API shutting down...")
