from fastapi import APIRouter, HTTPException, status
from app.models import ChatRequest, ChatResponse
from app.services.llm_service import LLMService
import logging

router = APIRouter(prefix="/api/v1", tags=["chat"])
llm_service = LLMService()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send messages to LLM and get response with performance metrics
    
    - **messages**: List of conversation messages with role and content
    - **max_tokens**: Maximum tokens in response (1-4096)
    - **temperature**: Randomness of output (0-2)
    - **provider**: LLM provider (openai/anthropic) - optional, uses default if not specified
    """
    try:
        result = await llm_service.chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            provider=request.provider
        )
        
        # Log performance metrics
        perf = result["performance"]
        logger.info(
            f"Chat completed | Provider: {result['provider']} | "
            f"Time: {perf.total_time_seconds}s | "
            f"Tokens/sec: {perf.tokens_per_second} | "
            f"Total tokens: {perf.total_tokens}"
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/chat/simple")
async def simple_chat(message: str, provider: str = None):
    """
    Simplified endpoint - send a single message and get response
    
    - **message**: Your message
    - **provider**: openai or anthropic (optional)
    """
    try:
        from app.models import Message
        
        result = await llm_service.chat_completion(
            messages=[Message(role="user", content=message)],
            provider=provider
        )
        
        return {
            "response": result["response"],
            "performance": result["performance"].dict()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/providers")
async def get_providers():
    """Get list of available LLM providers"""
    return {
        "available_providers": llm_service.get_available_providers(),
        "default_provider": llm_service.default_provider
    }
