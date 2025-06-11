import httpx
import json
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from models import CvInput, JdInput, CvMatchResult, ProjectInput, LanguageInput, AIAgentRecommendation
from ai_agent import match_cvs_with_agent
from ai_agent_controller import ai_agent_controller
from agent_integration_service import agent_integration_service
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
async def match_all_cvs(job_id: str, skill_filter: Optional[str] = None, use_ai_agents: bool = True):
    """
    Đánh giá tất cả CV trong hệ thống với JD và trả về kết quả kèm đề xuất hành động
    Enhanced with AI Agent system for optimized performance (1+2N → 1+2×ceil(N/batch_size) LLM calls)
    """
    try:
        headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

        # Get job data
        async with httpx.AsyncClient() as client:
            job_response = await client.get(f"http://localhost:8080/api/v1/job/{job_id}")
            job_response.raise_for_status()
            job_data = job_response.json()
            logger.info(f"JD response: {job_data}")
            description = job_data.get("data", {}).get("description", "") or "No job description provided"
            
            # Create enhanced JD input with job_id for agent system
            jd = JdInput(
                required_skills=job_data.get("data", {}).get("skillNames", []),
                required_experience=description,
                required_education="",
                required_certifications=[],
                text=description
            )
            # Add job_id for agent processing
            jd.job_id = job_id        # Get CV data with jobId parameter to filter evaluated CVs
        async with httpx.AsyncClient() as client:
            # Add jobId parameter to get CVs that haven't been evaluated for this job
            cv_response = await client.get(
                "http://localhost:8080/api/v1/cv/all", 
                headers=headers,
                params={"jobId": job_id}
            )
            cv_response.raise_for_status()
            cv_data = cv_response.json()["data"]
            logger.info(f"CV data count for job {job_id}: {len(cv_data)}")
            
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
                    phone=cv.get("info", {}).get("phone", "")
                ) for cv in cv_data
            ]
          # Check if no CVs to evaluate
        if not cvs:
            logger.info(f"No unevaluated CVs found for job {job_id}")
            return {
                "job_id": job_id,
                "total_candidates": 0,
                "results": [],
                "summary": {
                    "send_contact_email": 0,
                    "save_cv": 0,
                    "no_recommendation": 0
                },
                "message": "Không có CV nào chưa được đánh giá cho vị trí này. Tất cả CV đã được xử lý trước đó.",
                "processing_method": "none"
            }
        
        # Choose processing method based on use_ai_agents flag
        if use_ai_agents and len(cvs) > 3:
            logger.info(f"Using AI Agent system for {len(cvs)} CVs - optimized batch processing")
            results = await agent_integration_service.match_cvs_with_agents(
                cvs=cvs, 
                jd=jd, 
                use_batch_processing=True,
                batch_size=20
            )
            processing_method = "ai_agents_batch"
        else:
            logger.info(f"Using traditional processing for {len(cvs)} CVs")
            results = await match_cvs_with_agent(cvs, jd)
            # Add recommended actions for traditional processing
            for result in results:
                if result.score >= 80:
                    result.recommended_action = "send_contact_email"
                    result.action_reason = f"Ứng viên rất tiềm năng với {result.score}% độ phù hợp. Nên liên hệ ngay."
                elif result.score >= 50:
                    result.recommended_action = "save_cv"
                    result.action_reason = f"Ứng viên khá tiềm năng với {result.score}% độ phù hợp. Nên lưu lại cho tương lai."
                else:
                    result.recommended_action = None
                    result.action_reason = f"Ứng viên có {result.score}% độ phù hợp, chưa đạt tiêu chuẩn tối thiểu."            
                    processing_method = "traditional"
        
        # Save evaluation results to backend
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
            },            
            "processing_method": processing_method,
            "ai_agent_optimization": {
                "enabled": use_ai_agents and len(cvs) > 3,
                "batch_size": 20 if use_ai_agents else 1,
                "estimated_llm_calls_saved": max(0, len(cvs) * 2 - 2 * ((len(cvs) + 19) // 20)) if use_ai_agents and len(cvs) > 3 else 0
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

@app.post("/ai-agent/match-all/{job_id}")
async def ai_agent_match_all(job_id: str, batch_size: int = 20, priority: str = "normal"):
    """
    AI Agent enhanced endpoint for batch CV-JD matching with optimized performance
    Uses 1+2×ceil(N/batch_size) LLM calls instead of 1+2N calls
    """
    try:
        headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
          # Get job and CV data with jobId parameter for filtering
        async with httpx.AsyncClient() as client:
            job_response = await client.get(f"http://localhost:8080/api/v1/job/{job_id}")
            job_response.raise_for_status()
            job_data = job_response.json()
            
            # Add jobId parameter to get CVs that haven't been evaluated for this job
            cv_response = await client.get(
                "http://localhost:8080/api/v1/cv/all", 
                headers=headers,
                params={"jobId": job_id}
            )            
            cv_response.raise_for_status()
            cv_data = cv_response.json()["data"]
            
        if not cv_data:
            return {
                "job_id": job_id,
                "total_candidates": 0,
                "status": "no_candidates",
                "message": "No unevaluated CVs found for this job. All CVs may have been processed already."
            }
        
        cv_ids = [cv["id"] for cv in cv_data]
        
        # Submit batch job to AI Agent system
        batch_job_id = await agent_integration_service.batch_service.submit_batch_job(
            job_id=job_id,
            cv_ids=cv_ids,
            batch_size=batch_size,
            priority=priority
        )
        
        # Return job status for tracking
        return {
            "job_id": job_id,
            "batch_job_id": batch_job_id,
            "total_candidates": len(cv_ids),
            "batch_size": batch_size,
            "priority": priority,
            "status": "submitted",
            "estimated_llm_calls": 1 + 2 * ((len(cv_ids) + batch_size - 1) // batch_size),
            "optimization_ratio": f"{1 + 2 * len(cv_ids)} → {1 + 2 * ((len(cv_ids) + batch_size - 1) // batch_size)}",
            "message": "Batch job submitted successfully. Use /ai-agent/status/{batch_job_id} to track progress."
        }
        
    except Exception as e:
        logger.error(f"Error in AI agent batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Agent processing failed: {str(e)}")

@app.get("/ai-agent/status/{batch_job_id}")
async def get_batch_job_status(batch_job_id: str):
    """
    Get status of AI Agent batch processing job
    """
    try:
        status = await agent_integration_service.batch_service.get_job_status(batch_job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Batch job not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/ai-agent/result/{batch_job_id}")
async def get_batch_job_result(batch_job_id: str):
    """
    Get result of completed AI Agent batch processing job
    """
    try:
        result = await agent_integration_service.batch_service.get_job_result(batch_job_id)
        
        if not result:
            status = await agent_integration_service.batch_service.get_job_status(batch_job_id)
            if not status:
                raise HTTPException(status_code=404, detail="Batch job not found")
            elif status["status"] != "completed":
                raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {status['status']}")
            else:
                raise HTTPException(status_code=500, detail="Job completed but result not available")
        
        # Convert to API format
        api_results = []
        for match in result.results:
            api_results.append({
                "cv_id": match.cv_id,
                "score": match.overall_score,
                "explanation": match.explanation,
                "recommended_action": match.recommended_action,
                "action_reason": match.action_reason,
                "detailed_scores": {
                    "skill_match": match.skill_match_score,
                    "experience_match": match.experience_match_score,
                    "education_match": match.education_match_score,
                    "cultural_fit": match.cultural_fit_score,
                    "growth_potential": match.growth_potential_score
                },
                "skills_analysis": {
                    "matched": match.matched_skills,
                    "missing": match.missing_skills,
                    "transferable": match.transferable_skills
                },
                "confidence": match.confidence
            })
        
        return {
            "job_id": result.job_id,
            "total_candidates": result.total_cvs,
            "processed_candidates": result.processed_cvs,
            "failed_candidates": result.failed_cvs,
            "processing_time": result.processing_time,
            "results": api_results,
            "summary": result.summary,
            "errors": result.errors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job result: {str(e)}")

@app.get("/ai-agent/health")
async def ai_agent_health_check():
    """
    Health check for AI Agent system
    """
    try:
        health = await agent_integration_service.get_service_health()
        return health
        
    except Exception as e:
        logger.error(f"AI Agent health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/ai-agent/metrics")
async def ai_agent_metrics():
    """
    Get AI Agent system performance metrics
    """
    try:
        metrics = await agent_integration_service.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get AI Agent metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=8000, max_attempts=10):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("0.0.0.0", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")
    
    try:
        port = find_free_port(8000)
        print("🚀 Starting AI Agent Service...")
        print(f"📡 Service will be available at: http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"🔄 Health Check: http://localhost:{port}/ai-agent/health")
        print("=" * 60)
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            reload=False
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try stopping any existing servers on port 8000 or restart your terminal")
        exit(1)