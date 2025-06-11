import json
import re
import nltk
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
import os
import logging
from openai import AsyncOpenAI
from function_tools import ai_tools

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK data
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt data")
        nltk.download('punkt', quiet=True)

init_nltk()

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

class ExtractedData(BaseModel):
    skills: List[str] = []
    experience: str = ""
    education: str = ""
    certifications: List[str] = []
    projects: Optional[List[Dict[str, str]]] = None
    languages: Optional[List[Dict[str, str]]] = None
    years_experience: Optional[float] = None

async def extract_data(text: str, type: str) -> ExtractedData:
    if not text.strip():
        logger.warning(f"Empty {type} text provided, returning default ExtractedData")
        return ExtractedData(
            skills=[],
            experience="",
            education="",
            certifications=[],
            projects=[],
            languages=[],
            years_experience=None
        )
    
    prompt = (
        f"Extract the following from the {type} text in JSON format: "
        "skills (list of strings, include technical skills only, exclude language proficiency), "
        "experience (summary as a string, include years and roles), "
        "education (summary as a string, include degrees and fields), "
        "certifications (list of strings), "
        "projects (list of objects with project name and description, return empty list if none), "
        "languages (list of objects with language and proficiency level, return empty list if none). "
        "Return ONLY the JSON object, no explanatory text, markdown, or additional content."
    )
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured data from text. Always return valid JSON format without any additional text or markdown formatting."
                },
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nText: {text}"
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        text_content = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response for {type} extraction: {text_content}")
        
        # Clean up JSON content
        json_content = text_content.replace("```json\n", "").replace("\n```", "").strip()
        
        try:
            data = json.loads(json_content)
            
            # Extract years of experience using NLTK
            years = None
            tokens = word_tokenize(text.lower())            
            for i, token in enumerate(tokens):
                if token in ["year", "years", "yr", "yrs"]:
                    if i > 0 and re.match(r'^\d+\.?\d*$', tokens[i-1]):
                        years = float(tokens[i-1])
                        break
            
            data["years_experience"] = years
            data.setdefault("projects", [])
            data.setdefault("languages", [])
            
            for lang in data.get("languages", []):
                lang["level"] = lang.get("level", "") if lang.get("level") is not None else ""
                
            return ExtractedData(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in OpenAI response: {json_content}")
            raise ValueError(f"Invalid JSON in OpenAI response: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

async def calculate_relevance(cv_data: ExtractedData, jd_data: ExtractedData) -> Tuple[float, str]:
    prompt = (
        "You are an expert in evaluating CVs against Job Descriptions (JDs) for recruitment. Given a CV and a JD, perform a comprehensive evaluation to determine how well the CV matches the JD. Follow these steps:\n\n"
        "1. **Extract Requirements from JD**:\n"
        "   - Identify mandatory skills (technical skills like Java, Spring Boot, etc., listed in `skillNames` or mentioned in `description`).\n"
        "   - Identify required experience (years, roles, or specific domains mentioned in `description` or `experienceYear`).\n"
        "   - Identify required education (degrees, fields mentioned in `description`).\n"
        "   - Identify required certifications (if any).\n"
        "   - Identify required languages and proficiency levels (e.g., 'fluent in English' from `description`).\n"
        "   - Identify required project experience (specific project types or domains mentioned in `description`).\n\n"
        "2. **Extract Information from CV**:\n"
        "   - Identify skills from `skills` (technical and soft skills).\n"
        "   - Identify experience from `experience` (summary of years, roles, domains).\n"
        "   - Identify education from `education` (degrees, fields).\n"
        "   - Identify certifications from `certificates`.\n"
        "   - Identify languages and proficiency levels from `languages`.\n"
        "   - Identify project experience from `projects` (project names, descriptions).\n\n"
        "3. **Evaluate Match with Weighted Scoring**:\n"
        "   - Assign weights to each category:\n"
        "     - Skills: 40% (prioritize technical skills, partial match for related skills)\n"
        "     - Experience: 30% (consider years, roles, and domain relevance)\n"
        "     - Education: 10% (full match for exact degrees/fields, partial for related)\n"
        "     - Certifications: 5% (full match for listed certifications)\n"
        "     - Languages: 10% (full match for required languages with sufficient proficiency)\n"
        "     - Projects: 5% (match based on relevance to JD requirements)\n"
        "   - For each category:\n"
        "     - Calculate a match score (0-100) based on how well the CV meets the JD requirements.\n"
        "     - Consider semantic similarity (e.g., 'API development' in JD matches 'Built RESTful APIs' in CV projects).\n"
        "     - Account for partial matches (e.g., 2 years experience vs 3 years required gives partial score).\n"
        "   - Compute the final score as a weighted sum of category scores.\n\n"
        "4. **Provide Detailed Explanation**:\n"
        "   - For each category, explain the match score, including matched items, missing items, and partial matches.\n"
        "   - Highlight any inferred matches (e.g., skills inferred from projects or experience).\n\n"
        "**Input**:\n"
        "- **JD**:\n"
        "  - skillNames: {jd_skillNames}\n"
        "  - description: {jd_description}\n"
        "  - experienceYear: {jd_experienceYear}\n"
        "- **CV**:\n"
        "  - skills: {cv_skills}\n"
        "  - experience: {cv_experience}\n"
        "  - education: {cv_education}\n"
        "  - certificates: {cv_certificates}\n"
        "  - languages: {cv_languages}\n"
        "  - projects: {cv_projects}\n\n"
        "**Output**:\n"
        "Return a JSON object with:\n"
        "- score: float (0-100, overall match score)\n"
        "- explanation: string (detailed explanation of the match for each category)\n\n"
        "Example Output:\n"
        "```json\n"
        "{\"score\": 92.5, \"explanation\": \"Skills: Matched 4/5 mandatory skills (80%); Experience: CV 3 years vs JD 3 years (100% match); Education: Matched BS in Computer Science (100%); Certifications: Matched 0/0 certifications (100%); Languages: Matched English (Fluent) (100%); Projects: Matched e-learning project (100%)\"}\n"
        "```\n\n"
        "**Notes**:\n"
        "- Be strict with mandatory skills listed in `skillNames`.\n"
        "- Infer skills from CV `projects` or `experience` if not listed in `skills`.\n"
        "- For languages, match proficiency levels (e.g., 'Fluent' or 'Advanced' satisfies 'fluent in English').\n"
        "- Return only the JSON object, no additional text or markdown."
    )
    
    # Format the prompt with actual data
    formatted_prompt = prompt.replace("{jd_skillNames}", json.dumps(jd_data.skills)) \
                             .replace("{jd_description}", jd_data.experience) \
                             .replace("{jd_experienceYear}", str(jd_data.years_experience or "")) \
                             .replace("{cv_skills}", json.dumps(cv_data.skills)) \
                             .replace("{cv_experience}", json.dumps(cv_data.experience)) \
                             .replace("{cv_education}", json.dumps(cv_data.education)) \
                             .replace("{cv_certificates}", json.dumps(cv_data.certifications)) \
                             .replace("{cv_languages}", json.dumps(cv_data.languages)) \
                             .replace("{cv_projects}", json.dumps(cv_data.projects))
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR recruiter who evaluates CV-JD matches. Always return valid JSON format without any additional text or markdown formatting."
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        text_content = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response for relevance: {text_content}")
        
        # Clean up JSON content
        json_content = text_content.replace("```json\n", "").replace("\n```", "").strip()
        
        try:
            result = json.loads(json_content)
            score = result.get("score", 0.0)
            explanation = result.get("explanation", "")
            return score, explanation
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in OpenAI response: {json_content}")
            raise ValueError(f"Invalid JSON in OpenAI response: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

async def normalize_data(data: dict, type: str) -> ExtractedData:
    try:
        if type == "JD":
            text = data.get("text", "")
            skills = data.get("required_skills", [])
            experience = data.get("required_experience", "")
            education = data.get("required_education", "")
            certifications = data.get("required_certifications", [])
            
            extracted = await extract_data(text, type)
            languages = extracted.languages or []
            
            # Check for English requirement in JD text
            if "fluent in English".lower() in text.lower():
                languages.append({"language": "English", "level": "Fluent"})
                
            return ExtractedData(
                skills=skills or extracted.skills,
                experience=experience or extracted.experience,
                education=education or extracted.education,
                certifications=certifications or extracted.certifications,
                projects=extracted.projects or [],
                languages=languages,
                years_experience=extracted.years_experience
            )
        else:  # CV
            text = data.get("text", "")
            extracted = await extract_data(text, type)
            
            # Process languages with proper level handling
            languages = [
                {"language": l["language"], "level": l["level"] if l["level"] is not None else ""}
                for l in data.get("languages", extracted.languages or [])
            ]
            
            return ExtractedData(
                skills=data.get("skills", extracted.skills),
                experience=data.get("experience", extracted.experience),
                education=data.get("education", extracted.education),
                certifications=data.get("certifications", extracted.certifications),
                projects=data.get("projects", extracted.projects or []),
                languages=languages,
                years_experience=extracted.years_experience
            )
            
    except Exception as e:
        logger.error(f"Error normalizing {type} data: {str(e)}")
        raise ValueError(f"Invalid {type} data: {str(e)}")

# Function definitions for OpenAI function calling
FUNCTION_DEFINITIONS = [
    {
        "name": "get_cv_data",
        "description": "Retrieve CV data by ID or get all CVs if no ID provided",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV to retrieve. If not provided, returns all CVs"
                }
            }
        }
    },
    {
        "name": "get_job_data", 
        "description": "Retrieve job description data by job ID",
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
        "description": "Perform CV-JD matching analysis between a specific CV and job",
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
                    "reason": f"Ứng viên rất tiềm năng với {score}% độ phù hợp. Nên liên hệ ngay để không bỏ lỡ nhân tài.",
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
