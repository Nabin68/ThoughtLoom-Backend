# from pydantic import BaseModel
# from typing import Dict, Any, List

# class AnalyzeRequest(BaseModel):
#     reason: str
#     mcq_answers: Dict[str, Any]
#     additional_context: str

# class Insight(BaseModel):
#     title: str
#     explanation: str
#     next_steps: List[str]
#     caution: str

# class AnalyzeResponse(BaseModel):
#     summary: str
#     insights: List[Insight]


from pydantic import BaseModel, Field
from typing import Dict, Any, List

class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint"""
    reason: str = Field(..., description="Primary reason for seeking guidance")
    mcq_answers: Dict[str, Any] = Field(..., description="User's MCQ responses")
    additional_context: str = Field(..., description="User's own description of their situation")

class Insight(BaseModel):
    """Single insight with actionable guidance"""
    title: str = Field(..., description="Catchy, specific title for the insight")
    explanation: str = Field(..., description="Clear explanation of the issue (3-4 sentences)")
    next_steps: List[str] = Field(..., description="Specific, actionable steps")
    caution: str = Field(..., description="Warning about common pitfalls")

class AnalyzeResponse(BaseModel):
    """Response model with structured insights"""
    summary: str = Field(..., description="Brief emotional + practical understanding (2-3 sentences)")
    insights: List[Insight] = Field(..., description="List of key insights with actions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": "You're feeling stuck between family expectations and your own interests. This pressure is real and common in our context.",
                "insights": [
                    {
                        "title": "The Engineering Trap",
                        "explanation": "Your family pushes engineering because it feels safe, but safety at the cost of daily misery isn't wisdom. The real risk is spending years in a career that drains you.",
                        "next_steps": [
                            "List 3 fields you're genuinely curious about",
                            "Find 2 people in each field and ask about their daily work",
                            "Give yourself 2 weeks to research before making any commitments"
                        ],
                        "caution": "Don't romanticize creative fields either - they have their own challenges and require different kinds of discipline."
                    }
                ]
            }
        }