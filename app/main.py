import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.adaptive import router as adaptive_router
from app.api.completion import router as completion_router
from app.api.recommendation import router as recommendation_router

load_dotenv()

app = FastAPI(
    title="ThoughtLoom API",
    description="Weave clarity into your decisions",
    version="1.0.0"
)

# Comma-separated, e.g. ALLOWED_ORIGINS="https://thoughtloom.app,http://localhost:8080".
# The client sends no cookies or auth headers, so credentials stay off — which is also
# what makes the "*" default legal (browsers reject "*" alongside credentials).
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    # Authorization joins the list because every endpoint below /api carries the
    # caller's Supabase access token. Without it a browser build would fail CORS
    # preflight before the request left.
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(adaptive_router, prefix="/api", tags=["Conversation"])
app.include_router(recommendation_router, prefix="/api", tags=["Conversation"])
app.include_router(completion_router, prefix="/api", tags=["Conversation"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ThoughtLoom API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "adaptive_question": "/api/adaptive-question",
            "recommendation": "/api/recommendation",
            "follow_up": "/api/follow-up",
            "complete_chat": "/api/complete-chat",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ThoughtLoom"}
