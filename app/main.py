from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.analyze import router as analyze_router

app = FastAPI(
    title="ThoughtLoom API",
    description="Weave clarity into your decisions",
    version="1.0.0"
)

# CORS configuration - Allow Flutter web to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ CRITICAL: Include the analyze router
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ThoughtLoom API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ThoughtLoom"}