# from fastapi import APIRouter
# from app.schemas.request_response import AnalyzeRequest
# from app.services.reasoning_engine import generate_insight

# router = APIRouter(prefix="/api")

# @router.post("/analyze")
# def analyze(payload: AnalyzeRequest):
#     result = generate_insight(payload.dict())
#     return {"raw_output": result}


from fastapi import APIRouter, HTTPException
from app.schemas.request_response import AnalyzeRequest, AnalyzeResponse
from app.services.reasoning_engine import generate_insight

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze user input and generate personalized insights.
    
    Returns structured JSON with summary and insights.
    """
    try:
        payload = {
            "reason": request.reason,
            "mcq_answers": request.mcq_answers,
            "additional_context": request.additional_context,
        }
        
        # Generate insights
        result = generate_insight(payload)
        
        # Validate structure
        if not isinstance(result, dict):
            raise ValueError("Invalid response format from LLM")
        
        if "summary" not in result or "insights" not in result:
            raise ValueError("Missing required fields in response")
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating insights: {str(e)}"
        )