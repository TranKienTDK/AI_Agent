import httpx
import json
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from models import CvInput, JdInput, CvMatchResult, ProjectInput, LanguageInput
from agent_integration_service import agent_integration_service
import uvicorn
from dotenv import load_dotenv
import os

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

async def match_cvs_with_agent(cvs: List[CvInput], jd: JdInput) -> List[CvMatchResult]:
    """
    Fallback function - sử dụng agent_integration_service
    """
    try:
        return await agent_integration_service.match_cvs_with_agents(
            cvs=cvs, 
            jd=jd, 
            use_batch_processing=False
        )
    except Exception as e:
        logger.error(f"Fallback matching failed: {str(e)}")
        # Return empty results with error
        return [CvMatchResult(
            cv_id=cv.cv_id,
            score=0.0,
            explanation=f"Error: {str(e)}",
            email=cv.email,
            phone=cv.phone
        ) for cv in cvs]

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
                "estimated_llm_calls_saved": max(0, len(cvs) * 2 - 2 * ((len(cvs) + 19) // 20)) if use_ai_agents and len(cvs) > 3 else 0            }
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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