import httpx
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from models import CvInput, JdInput, CvMatchResult, ProjectInput, LanguageInput, AIAgentRecommendation
from ai_agent import match_cvs_with_agent
from ai_agent_controller import ai_agent_controller
import uvicorn
from dotenv import load_dotenv
import os
from cv_evaluation_system import (
    EvaluationRequest, EvaluationResult,
    cv_evaluation_system
)

load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/match", response_model=List[CvMatchResult])
async def match_cvs(cvs: List[CvInput], jd: JdInput):
    try:
        results = await match_cvs_with_agent(cvs, jd)
        return results
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/match-all/{job_id}")
async def match_all_cvs(job_id: str, skill_filter: Optional[str] = None):
    """
    Đánh giá tất cả CV trong hệ thống với JD và trả về kết quả kèm đề xuất hành động
    """
    try:
        headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

        async with httpx.AsyncClient() as client:
            job_response = await client.get(f"http://localhost:8080/api/v1/job/{job_id}")
            job_response.raise_for_status()
            job_data = job_response.json()
            logger.info(f"JD response: {job_data}")
            description = job_data.get("data", {}).get("description", "") or "No job description provided"
            jd = JdInput(
                required_skills=job_data.get("data", {}).get("skillNames", []),
                required_experience=description,
                required_education="",
                required_certifications=[],
                text=description
            )

        async with httpx.AsyncClient() as client:
            cv_response = await client.get("http://localhost:8080/api/v1/cv/all", headers=headers)
            cv_response.raise_for_status()
            cv_data = cv_response.json()["data"]
            logger.info(f"CV data: {cv_data}")
            cvs = [
                CvInput(
                    cv_id=cv["id"],
                    skills=[skill["name"] for skill in cv.get("skills", [])],
                    experience="; ".join([exp["description"] for exp in cv.get("experiences", [])]),
                    education="; ".join([edu["field"] + ": " + edu.get("description", "") for edu in cv.get("educations", [])]),
                    certifications=[cert["certificate"] for cert in cv.get("certifications", [])],
                    projects=[
                        ProjectInput(
                            project=p["project"],
                            description=p.get("description", ""),
                            start_date=p.get("startDate", ""),
                            end_date=p.get("endDate", "")
                        ) for p in cv.get("projects", []) if p is not None and p.get("project") is not None and isinstance(p.get("project"), str)
                    ],
                    languages=[
                        LanguageInput(
                            language=l["language"],
                            level=l.get("level", "") if l.get("level") is not None else ""
                        ) for l in cv.get("languages", []) if l is not None and l.get("language") is not None and isinstance(l.get("language"), str)
                    ],
                    text=cv.get("profile", "") + " " + cv.get("additionalInfo", ""),
                    email=cv.get("info", {}).get("email", ""),
                    phone=cv.get("info", {}).get("phone", "")                ) for cv in cv_data
            ]
        
        # Kiểm tra nếu không có CV nào để đánh giá
        if not cvs:
            logger.info(f"No CVs found for evaluation with job {job_id}")
            return {
                "job_id": job_id,
                "total_candidates": 0,
                "results": [],
                "summary": {
                    "send_contact_email": 0,
                    "save_cv": 0,
                    "no_recommendation": 0
                },
                "message": "Không có CV nào trong hệ thống để đánh giá cho vị trí này."
            }
        
        # Thực hiện matching và tạo recommendations
        results = await match_cvs_with_agent(cvs, jd)
          # Thêm recommended actions cho mỗi result
        for result in results:
            # Tạo recommendations dựa trên score - chỉ 2 action types
            if result.score >= 80:
                result.recommended_action = "send_contact_email"
                result.action_reason = f"Ứng viên rất tiềm năng với {result.score}% độ phù hợp. Nên liên hệ ngay."
            elif result.score >= 50:
                result.recommended_action = "save_cv"
                result.action_reason = f"Ứng viên khá tiềm năng với {result.score}% độ phù hợp. Nên lưu lại cho tương lai."
            else:
                result.recommended_action = None  # No action for low scores
                result.action_reason = f"Ứng viên có {result.score}% độ phù hợp, chưa đạt tiêu chuẩn tối thiểu."        # Lưu evaluation results vào backend
        async with httpx.AsyncClient() as client:
            for result in results:
                # Extract skill names as strings for the evaluation
                cv_skills = []
                cv_info = next((cv for cv in cv_data if cv["id"] == result.cv_id), None)
                if cv_info and cv_info.get("skills"):
                    cv_skills = [skill["name"] for skill in cv_info["skills"] if isinstance(skill, dict) and skill.get("name")]
                
                evaluation = {
                    "cvId": result.cv_id,
                    "jobId": job_id,
                    "score": result.score,
                    "explanation": result.explanation,
                    "skills": cv_skills,
                    "feedback": None,
                    "recommendedAction": result.recommended_action,
                    "actionReason": result.action_reason
                }
                try:
                    await client.post("http://localhost:8080/api/v1/evaluations", json=evaluation, headers=headers)
                except Exception as e:
                    logger.error(f"Failed to save evaluation for CV {result.cv_id}: {str(e)}")

        return {
            "job_id": job_id,
            "total_candidates": len(results),
            "results": results,
            "summary": {
                "send_contact_email": len([r for r in results if r.recommended_action == "send_contact_email"]),
                "save_cv": len([r for r in results if r.recommended_action == "save_cv"]),
                "no_recommendation": len([r for r in results if r.recommended_action is None])
            }
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/evaluation/batch", response_model=EvaluationResult)
async def batch_evaluation(request: EvaluationRequest):
    """
    Đánh giá hàng loạt CV cho một job position
    """
    try:
        result = await cv_evaluation_system.evaluate_candidates(request)
        return result
    except Exception as e:
        logger.error(f"Error in batch evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/evaluation/single/{cv_id}/{job_id}")
async def single_evaluation(cv_id: str, job_id: str, save_result: bool = True):
    """
    Đánh giá một CV cụ thể cho một job
    """
    try:
        result = await cv_evaluation_system.evaluate_single_cv(cv_id, job_id, save_result)
        return result
    except Exception as e:
        logger.error(f"Error in single evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/evaluation/top-candidates/{job_id}")
async def get_top_candidates(job_id: str, top_n: int = 5):
    """
    Lấy top N ứng viên phù hợp nhất cho một job
    """
    try:
        result = await cv_evaluation_system.get_top_candidates(job_id, top_n)
        return result
    except Exception as e:
        logger.error(f"Error getting top candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/evaluation/status/{job_id}")
async def get_evaluation_status(job_id: str):
    """
    Kiểm tra trạng thái đánh giá cho một job
    """
    try:
        # Có thể mở rộng để tracking evaluation progress
        return {
            "job_id": job_id,
            "status": "ready",
            "message": "Evaluation system is ready"
        }
    except Exception as e:
        logger.error(f"Error checking evaluation status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ai-agent/recommend/{cv_id}/{job_id}", response_model=AIAgentRecommendation)
async def get_ai_recommendations(cv_id: str, job_id: str):
    """
    Get AI Agent recommendations for a specific CV-Job match
    HR can review recommendations and decide which actions to take
    """
    try:
        recommendation = await ai_agent_controller.get_recommendations_for_match(cv_id, job_id)
        return recommendation
    except Exception as e:
        logger.error(f"Error getting AI recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ai-agent/execute-action")
async def execute_recommended_action(
    action_type: str,
    cv_id: str, 
    job_id: str,
    score: float = None,
    explanation: str = None,
    skills: List[str] = None
):
    """
    Execute a specific action recommended by AI Agent
    This allows HR to selectively execute only approved actions
    """
    try:
        if action_type == "save_evaluation":
            if not all([score is not None, explanation]):
                raise HTTPException(status_code=400, detail="Score and explanation required for save_evaluation")
                
            # Execute save_evaluation 
            from function_tools import ai_tools
            result = await ai_tools.save_evaluation(cv_id, job_id, score, explanation, skills)
            return {
                "success": True,
                "action_executed": action_type,
                "result": result,
                "message": f"Successfully executed {action_type} for CV {cv_id} and Job {job_id}"
            }
        else:
            return {
                "success": True,
                "action_executed": action_type,
                "message": f"Action '{action_type}' noted. Please handle this action manually in your HR system.",
                "instructions": {
                    "approve_candidate": "Contact candidate and proceed with hiring process",
                    "request_interview": "Schedule interview with candidate", 
                    "request_more_info": "Request additional information from candidate",
                    "save_for_later": "Add candidate to talent pool for future opportunities",
                    "reject_candidate": "Send rejection notification to candidate"
                }.get(action_type, "Handle this action according to your HR procedures")
            }
            
    except Exception as e:
        logger.error(f"Error executing action {action_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)