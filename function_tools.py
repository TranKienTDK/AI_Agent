import httpx
import json
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import os
from models import CvInput, JdInput, ProjectInput, LanguageInput

load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

logger = logging.getLogger(__name__)

class AIAgentTools:
    """
    Function tools cho AI Agent để tương tác với hệ thống backend
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8080/api/v1"
        self.headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
    
    async def get_cv_data(self, cv_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy dữ liệu CV theo ID hoặc tất cả CV
        
        Args:
            cv_id: ID của CV cần lấy. Nếu None thì lấy tất cả CV
            
        Returns:
            Dict chứa thông tin CV
        """
        try:
            async with httpx.AsyncClient() as client:
                if cv_id:
                    url = f"{self.base_url}/cv/{cv_id}"
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()
                    cv_data = response.json()["data"]
                    
                    # Convert single CV to CvInput format
                    cv = self._convert_to_cv_input(cv_data)
                    return {
                        "success": True,
                        "data": cv.dict(),
                        "message": f"Successfully retrieved CV {cv_id}"
                    }
                else:
                    url = f"{self.base_url}/cv/all"
                    response = await client.get(url, headers=self.headers)
                    response.raise_for_status()
                    cv_list = response.json()["data"]
                    
                    # Convert all CVs to CvInput format
                    cvs = [self._convert_to_cv_input(cv) for cv in cv_list]
                    return {
                        "success": True,
                        "data": [cv.dict() for cv in cvs],
                        "count": len(cvs),
                        "message": f"Successfully retrieved {len(cvs)} CVs"
                    }
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting CV data: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "message": "Failed to retrieve CV data"
            }
        except Exception as e:
            logger.error(f"Error getting CV data: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve CV data"
            }
    
    async def get_job_data(self, job_id: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu Job Description theo ID
        
        Args:
            job_id: ID của job cần lấy
            
        Returns:
            Dict chứa thông tin JD
        """
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/job/{job_id}"
                response = await client.get(url)
                response.raise_for_status()
                job_data = response.json()["data"]
                
                # Convert to JdInput format
                description = job_data.get("description", "") or "No job description provided"
                jd = JdInput(
                    required_skills=job_data.get("skillNames", []),
                    required_experience=description,
                    required_education="",
                    required_certifications=[],
                    text=description
                )
                
                return {
                    "success": True,
                    "data": jd.dict(),
                    "job_info": {
                        "id": job_id,
                        "title": job_data.get("title", ""),
                        "company": job_data.get("company", ""),
                        "description": description
                    },
                    "message": f"Successfully retrieved job {job_id}"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting job data: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "message": "Failed to retrieve job data"
            }
        except Exception as e:
            logger.error(f"Error getting job data: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve job data"
            }
    
    async def save_evaluation(self, cv_id: str, job_id: str, score: float, explanation: str, 
                            skills: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Lưu kết quả đánh giá CV-JD matching
        
        Args:
            cv_id: ID của CV
            job_id: ID của Job
            score: Điểm đánh giá (0-100)
            explanation: Giải thích chi tiết
            skills: Danh sách skills (optional)
            
        Returns:
            Dict chứa kết quả lưu
        """
        try:
            evaluation_data = {
                "cvId": cv_id,
                "jobId": job_id,
                "score": score,
                "explanation": explanation,
                "skills": skills or [],
                "feedback": None
            }
            
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/evaluations"
                response = await client.post(url, json=evaluation_data, headers=self.headers)
                response.raise_for_status()
                
                return {
                    "success": True,
                    "data": evaluation_data,
                    "message": f"Successfully saved evaluation for CV {cv_id} and Job {job_id}"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when saving evaluation: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "message": "Failed to save evaluation"
            }
        except Exception as e:
            logger.error(f"Error saving evaluation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save evaluation"
            }
    
    async def match_cv_with_job(self, cv_id: str, job_id: str) -> Dict[str, Any]:
        """
        Thực hiện matching giữa CV và JD
        
        Args:
            cv_id: ID của CV
            job_id: ID của Job
            
        Returns:
            Dict chứa kết quả matching
        """
        try:
            # Import here to avoid circular import
            from ai_agent import match_cvs_with_agent
            
            # Get CV data
            cv_result = await self.get_cv_data(cv_id)
            if not cv_result["success"]:
                return cv_result
            
            # Get Job data
            job_result = await self.get_job_data(job_id)
            if not job_result["success"]:
                return job_result
            
            # Convert back to model objects
            cv_data = CvInput(**cv_result["data"])
            jd_data = JdInput(**job_result["data"])
            
            # Perform matching
            results = await match_cvs_with_agent([cv_data], jd_data)
            
            if results:
                result = results[0]
                return {
                    "success": True,
                    "data": {
                        "cv_id": result.cv_id,
                        "score": result.score,
                        "explanation": result.explanation,
                        "email": result.email,
                        "phone": result.phone
                    },
                    "cv_info": cv_result.get("data", {}),
                    "job_info": job_result.get("job_info", {}),
                    "message": f"Successfully matched CV {cv_id} with Job {job_id}"
                }
            else:
                return {
                    "success": False,
                    "error": "No matching results returned",
                    "message": "Failed to match CV with Job"
                }
                
        except Exception as e:
            logger.error(f"Error matching CV with job: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to match CV with Job"
            }
    
    async def get_raw_cv_data(self, cv_id: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu CV thô (raw) từ API - dùng cho AI Agents
        
        Args:
            cv_id: ID của CV cần lấy
            
        Returns:
            Dict chứa raw CV data từ API
        """
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/cv/{cv_id}"
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                cv_data = response.json()["data"]
                
                return {
                    "success": True,
                    "data": cv_data,  # Return raw API data, not converted
                    "message": f"Successfully retrieved raw CV {cv_id}"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting CV {cv_id}: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting CV {cv_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    def _convert_to_cv_input(self, cv_data: dict) -> CvInput:
        """
        Convert API CV data to CvInput model
        """
        return CvInput(
            cv_id=cv_data["id"],
            skills=[skill["name"] for skill in cv_data.get("skills", [])],
            experience="; ".join([exp["description"] for exp in cv_data.get("experiences", [])]),
            education="; ".join([
                edu["field"] + ": " + edu.get("description", "") 
                for edu in cv_data.get("educations", [])
            ]),
            certifications=[cert["certificate"] for cert in cv_data.get("certifications", [])],
            projects=[
                ProjectInput(
                    project=p["project"],
                    description=p.get("description", ""),
                    start_date=p.get("startDate", ""),
                    end_date=p.get("endDate", "")
                ) for p in cv_data.get("projects", []) 
                if p and p.get("project") and isinstance(p.get("project"), str)
            ],
            languages=[
                LanguageInput(
                    language=l["language"],
                    level=l.get("level", "") if l.get("level") is not None else ""
                ) for l in cv_data.get("languages", []) 
                if l and l.get("language") and isinstance(l.get("language"), str)
            ],
            text=cv_data.get("profile", "") + " " + cv_data.get("additionalInfo", ""),
            email=cv_data.get("info", {}).get("email", ""),
            phone=cv_data.get("info", {}).get("phone", "")
        )

# Global instance
ai_tools = AIAgentTools()
