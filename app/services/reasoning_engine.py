# from app.core.llm import llm
# from app.prompts.reasoning_prompt import PROMPT

# def generate_insight(payload: dict) -> str:
#     chain = PROMPT | llm
#     return chain.invoke({
#         "reason": payload["reason"],
#         "mcq_answers": payload["mcq_answers"],
#         "additional_context": payload["additional_context"],
#     }).content


import json
from app.core.llm import llm
from app.prompts.reasoning_prompt import PROMPT

def generate_insight(payload: dict) -> dict:
    """
    Generate structured insights from user input.
    Returns a dictionary with 'summary' and 'insights' keys.
    """
    chain = PROMPT | llm
    
    response = chain.invoke({
        "reason": payload["reason"],
        "mcq_answers": payload["mcq_answers"],
        "additional_context": payload["additional_context"],
    })
    
    raw_content = response.content.strip()
    
    # Try to parse as JSON
    try:
        # Remove markdown code blocks if present
        if raw_content.startswith("```"):
            # Find the first { and last }
            start_idx = raw_content.find("{")
            end_idx = raw_content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                raw_content = raw_content[start_idx:end_idx + 1]
        
        parsed_result = json.loads(raw_content)
        return parsed_result
    
    except json.JSONDecodeError:
        # Fallback: return raw content in a structured format
        return {
            "summary": "We're analyzing your situation and preparing insights.",
            "insights": [
                {
                    "title": "Analysis in Progress",
                    "explanation": raw_content[:200] + "..." if len(raw_content) > 200 else raw_content,
                    "next_steps": [
                        "Review the insights carefully",
                        "Take time to reflect on what resonates",
                        "Start with one small action"
                    ],
                    "caution": "Remember, these are suggestions to consider, not rules to follow blindly."
                }
            ]
        }