from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.ai_analyzer.agentMain import router as ai_analyzer_router
from app.r2r.routes import router as r2r_router
from app.auth.routes import router as auth_router
from app.streaming import streaming_router, streaming_engine
import uvicorn
import os

app = FastAPI(title="Standalone Finance AI Chatbot & Analyzer API")

# Add CORS middleware to allow requests from frontend (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(ai_analyzer_router, prefix="/ai-analyzer", tags=["ai-analyzer"])
app.include_router(r2r_router)
app.include_router(streaming_router)

@app.on_event("startup")
async def startup_event():
    await streaming_engine.start()

@app.on_event("shutdown")
async def shutdown_event():
    await streaming_engine.stop()

@app.get("/")
async def root():
    return {"message": "Standalone Finance AI Chatbot & Analyzer API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
