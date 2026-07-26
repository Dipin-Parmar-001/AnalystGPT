from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.api.health import router as health_router
from backend.app.parser import parse_and_save_csv

try:
    import logfire
    from agents.logger import setup_logfire
    setup_logfire(service_name="analyst-gpt-api")
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False

app = FastAPI(
    title="AnalystGPT API",
    description="Backend API for AnalystGPT",
    version="1.0.0",
)

if HAS_LOGFIRE:
    try:
        logfire.instrument_fastapi(app)
    except Exception as e:
        print(f"[Logfire Notice] FastAPI instrumentation skipped: {e}")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and parse a CSV file.
    """
    file_path = parse_and_save_csv(file)
    return {
        "status": "success",
        "message": "File parsed and saved successfully",
        "file_path": file_path
    }

@app.get("/")
async def root():
    """
    Root endpoint returning welcome message.
    """
    return {
        "message": "Welcome to AnalystGPT API",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Start the server when executed directly
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
