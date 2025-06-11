import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from function_tools import ai_tools
from openai_service import normalize_data, calculate_relevance
from models import CvInput, JdInput, CvMatchResult

logger = logging.getLogger(__name__)

class EvaluationRequest(BaseModel):
    job_id: str
    cv_ids: Optional[List[str]] = None  # Nếu None thì đánh giá tất cả CV
    save_results: bool = True
    top_n: Optional[int] = None  # Lấy top N candidates

class EvaluationResult(BaseModel):
    job_id: str
    total_cvs: int
    evaluated_cvs: int
    top_candidates: List[CvMatchResult]
    all_results: List[CvMatchResult]
    execution_time: float
    success: bool
    error_message: Optional[str] = None

class CVEvaluationSystem:
    """
    Hệ thống đánh giá CV-JD tự động không cần chat
    """
    
    def __init__(self):
        self.name = "CV-JD Evaluation System"
    
    async def evaluate_candidates(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Đánh giá ứng viên cho một vị trí công việc
        
        Args:
            request: EvaluationRequest chứa job_id và các CV cần đánh giá
            
        Returns:
            EvaluationResult với kết quả đánh giá
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting evaluation for job {request.job_id}")
            
            # 1. Lấy thông tin Job Description
            job_result = await ai_tools.get_job_data(request.job_id)
            if not job_result["success"]:
                return EvaluationResult(
                    job_id=request.job_id,
                    total_cvs=0,
                    evaluated_cvs=0,
                    top_candidates=[],
                    all_results=[],
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Failed to get job data: {job_result.get('error', 'Unknown error')}"
                )
            
            jd_data = JdInput(**job_result["data"])
            
            # 2. Lấy CV data
            if request.cv_ids:
                # Lấy CV theo danh sách ID
                cv_results = []
                for cv_id in request.cv_ids:
                    cv_result = await ai_tools.get_cv_data(cv_id)
                    if cv_result["success"]:
                        cv_results.append(cv_result["data"])
                    else:
                        logger.warning(f"Failed to get CV {cv_id}: {cv_result.get('error')}")
            else:
                # Lấy tất cả CV
                all_cv_result = await ai_tools.get_cv_data()
                if not all_cv_result["success"]:
                    return EvaluationResult(
                        job_id=request.job_id,
                        total_cvs=0,
                        evaluated_cvs=0,
                        top_candidates=[],
                        all_results=[],
                        execution_time=time.time() - start_time,
                        success=False,
                        error_message=f"Failed to get CV data: {all_cv_result.get('error', 'Unknown error')}"
                    )
                cv_results = all_cv_result["data"]
            
            total_cvs = len(cv_results)
            logger.info(f"Evaluating {total_cvs} CVs for job {request.job_id}")
            
            # 3. Thực hiện đánh giá từng CV
            all_results = []
            evaluated_count = 0
            
            for cv_data in cv_results:
                try:
                    cv_input = CvInput(**cv_data)
                    
                    # Normalize data
                    cv_normalized = await normalize_data(cv_data, "CV")
                    jd_normalized = await normalize_data(jd_data.dict(), "JD")
                    
                    # Calculate relevance
                    score, explanation = await calculate_relevance(cv_normalized, jd_normalized)
                    
                    # Create result
                    result = CvMatchResult(
                        cv_id=cv_input.cv_id,
                        score=score,
                        explanation=explanation,
                        email=cv_input.email,
                        phone=cv_input.phone
                    )
                    
                    all_results.append(result)
                    evaluated_count += 1
                    
                    # Save evaluation if requested
                    if request.save_results:
                        await ai_tools.save_evaluation(
                            cv_id=cv_input.cv_id,
                            job_id=request.job_id,
                            score=score,
                            explanation=explanation,
                            skills=cv_input.skills
                        )
                    
                    logger.info(f"Evaluated CV {cv_input.cv_id}: {score:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating CV {cv_data.get('cv_id', 'unknown')}: {str(e)}")
                    continue
            
            # 4. Sắp xếp theo điểm số
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # 5. Lấy top candidates
            top_candidates = all_results[:request.top_n] if request.top_n else all_results
            
            execution_time = time.time() - start_time
            
            logger.info(f"Evaluation completed: {evaluated_count}/{total_cvs} CVs in {execution_time:.2f}s")
            
            return EvaluationResult(
                job_id=request.job_id,
                total_cvs=total_cvs,
                evaluated_cvs=evaluated_count,
                top_candidates=top_candidates,
                all_results=all_results,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in evaluation system: {str(e)}")
            return EvaluationResult(
                job_id=request.job_id,
                total_cvs=0,
                evaluated_cvs=0,
                top_candidates=[],
                all_results=[],
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def evaluate_single_cv(self, cv_id: str, job_id: str, save_result: bool = True) -> Dict[str, Any]:
        """
        Đánh giá một CV cụ thể cho một job
        
        Args:
            cv_id: ID của CV
            job_id: ID của job
            save_result: Có lưu kết quả không
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        try:
            request = EvaluationRequest(
                job_id=job_id,
                cv_ids=[cv_id],
                save_results=save_result,
                top_n=1
            )
            
            result = await self.evaluate_candidates(request)
            
            if result.success and result.top_candidates:
                candidate = result.top_candidates[0]
                return {
                    "success": True,
                    "cv_id": candidate.cv_id,
                    "job_id": job_id,
                    "score": candidate.score,
                    "explanation": candidate.explanation,
                    "email": candidate.email,
                    "phone": candidate.phone,
                    "execution_time": result.execution_time
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message or "No evaluation result",
                    "cv_id": cv_id,
                    "job_id": job_id
                }
                
        except Exception as e:
            logger.error(f"Error evaluating single CV {cv_id} for job {job_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "cv_id": cv_id,
                "job_id": job_id
            }
    
    async def get_top_candidates(self, job_id: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Lấy top N ứng viên phù hợp nhất cho một công việc
        
        Args:
            job_id: ID của job
            top_n: Số lượng ứng viên top
            
        Returns:
            Dict chứa danh sách top candidates
        """
        try:
            request = EvaluationRequest(
                job_id=job_id,
                cv_ids=None,  # Đánh giá tất cả CV
                save_results=True,
                top_n=top_n
            )
            
            result = await self.evaluate_candidates(request)
            
            if result.success:
                return {
                    "success": True,
                    "job_id": job_id,
                    "total_cvs": result.total_cvs,
                    "evaluated_cvs": result.evaluated_cvs,
                    "top_candidates": [
                        {
                            "cv_id": c.cv_id,
                            "score": c.score,
                            "explanation": c.explanation,
                            "email": c.email,
                            "phone": c.phone
                        } for c in result.top_candidates
                    ],
                    "execution_time": result.execution_time
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message,
                    "job_id": job_id
                }
                
        except Exception as e:
            logger.error(f"Error getting top candidates for job {job_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id
            }

# Global evaluation system instance
cv_evaluation_system = CVEvaluationSystem()
