from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {
        "status": "healthy",
        "details": {
            "service": "AnalystGPT Backend"
        }
    }
