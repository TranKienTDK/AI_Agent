import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from function_tools import ai_tools

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Function definitions for AI Agent function calling
FUNCTION_DEFINITIONS = [
    {
        "name": "get_cv_data",
        "description": "Get CV data by ID from the system",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV to retrieve"
                }
            },
            "required": ["cv_id"]
        }
    },
    {
        "name": "get_job_data",
        "description": "Get job description data by ID from the system",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job to retrieve"
                }
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "match_cv_with_job",
        "description": "Match a specific CV with a job and return compatibility score and analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV to match"
                },
                "job_id": {
                    "type": "string", 
                    "description": "The ID of the job to match against"
                }
            },
            "required": ["cv_id", "job_id"]
        }
    },
    {
        "name": "recommend_actions",
        "description": "Generate HR action recommendations based on CV-JD matching results. Use this instead of automatically saving evaluations.",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV"
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job"
                },
                "score": {
                    "type": "number",
                    "description": "The matching score (0-100)"
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of the matching result"
                }
            },
            "required": ["cv_id", "job_id", "score", "explanation"]
        }
    },
    {
        "name": "save_evaluation",
        "description": "Save CV-JD evaluation results to the system. Only use when HR explicitly requests to save results.",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV"
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job"
                },
                "score": {
                    "type": "number",
                    "description": "The matching score (0-100)"
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of the matching result"
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of skills (optional)"
                }
            },
            "required": ["cv_id", "job_id", "score", "explanation"]
        }
    }
]

async def execute_function(function_name: str, arguments: dict) -> dict:
    """
    Execute a function call from the AI agent
    
    Args:
        function_name: Name of the function to call
        arguments: Arguments to pass to the function
        
    Returns:
        Result of the function execution
    """
    try:        
        if function_name == "get_cv_data":
            cv_id = arguments.get("cv_id")
            return await ai_tools.get_cv_data(cv_id)
        elif function_name == "get_job_data":
            job_id = arguments["job_id"]
            return await ai_tools.get_job_data(job_id)
            
        elif function_name == "match_cv_with_job":
            cv_id = arguments["cv_id"]
            job_id = arguments["job_id"]
            return await ai_tools.match_cv_with_job(cv_id, job_id)
            
        elif function_name == "recommend_actions":
            score = arguments["score"]
            cv_id = arguments["cv_id"]
            job_id = arguments["job_id"]
            explanation = arguments["explanation"]
            
            recommendations = []
            
            if score >= 70:
                # Ứng viên rất tiềm năng - gửi email liên hệ
                recommendations.append({
                    "action": "send_contact_email",
                    "priority": "high",
                    "reason": f"Ứng viên rất tiềm năng với {score}% độ phù hợp. Nên liên hệ ngay để sắp xếp phỏng vấn.",
                    "next_steps": ["Gửi email liên hệ trong vòng 24 giờ", "Sắp xếp cuộc gọi điện thoại sơ bộ", "Chuẩn bị câu hỏi phỏng vấn"]
                })
            elif score >= 50:
                # Ứng viên khá tiềm năng - lưu CV
                recommendations.append({
                    "action": "save_cv",
                    "priority": "medium",
                    "reason": f"Ứng viên khá tiềm năng với {score}% độ phù hợp. Nên lưu lại để xem xét cho các vị trí tương lai.",
                    "next_steps": ["Lưu CV vào cơ sở dữ liệu nhân tài", "Đánh dấu cho các vị trí tương tự trong tương lai", "Theo dõi sự phát triển kỹ năng của ứng viên"]
                })
            # Note: No specific action for scores < 50%, but evaluation data is still available
            
            # Always suggest saving evaluation as an option
            recommendations.append({
                "action": "save_evaluation",
                "priority": "medium",
                "reason": "Save evaluation results for future reference and reporting",
                "next_steps": ["Save to system database when HR approves"]
            })
            
            return {
                "success": True,
                "cv_id": cv_id,
                "job_id": job_id,
                "score": score,
                "explanation": explanation,
                "recommended_actions": recommendations,
                "message": "Action recommendations generated. HR should review and decide which actions to take."
            }
            
        elif function_name == "save_evaluation":
            return await ai_tools.save_evaluation(
                cv_id=arguments["cv_id"],
                job_id=arguments["job_id"],
                score=arguments["score"],
                explanation=arguments["explanation"],
                skills=arguments.get("skills")
            )
        else:
            return {
                "success": False,
                "error": f"Unknown function: {function_name}",
                "message": "Function not found"
            }
            
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to execute {function_name}"
        }

async def ai_agent_chat(messages: List[dict], use_functions: bool = True) -> dict:
    """
    AI Agent chat with function calling capabilities
    
    Args:
        messages: List of chat messages
        use_functions: Whether to enable function calling
        
    Returns:
        AI response with potential function calls
    """
    try:
        # Prepare the request
        request_params = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 3000
        }
        
        # Add functions if enabled
        if use_functions:
            request_params["tools"] = [
                {"type": "function", "function": func} for func in FUNCTION_DEFINITIONS
            ]
            request_params["tool_choice"] = "auto"
        
        response = await client.chat.completions.create(**request_params)
        
        message = response.choices[0].message
        
        # Handle function calls
        if message.tool_calls:
            function_results = []
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"Executing function: {function_name} with args: {function_args}")
                
                # Execute the function
                result = await execute_function(function_name, function_args)
                
                function_results.append({
                    "tool_call_id": tool_call.id,
                    "function_name": function_name,
                    "result": result
                })
                
                # Add function result to messages for context
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            # Get final response after function execution
            final_response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            return {
                "success": True,
                "message": final_response.choices[0].message.content,
                "function_calls": function_results,
                "has_function_calls": True
            }
        else:
            return {
                "success": True,
                "message": message.content,
                "function_calls": [],
                "has_function_calls": False
            }
            
    except Exception as e:
        logger.error(f"Error in AI agent chat: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to process AI agent request"
        }
