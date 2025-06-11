"""
AI Agent Integration Service - Wrapper for integrating AI Agent system with existing API
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from agents import (
    OrchestratorAgent, BatchProcessingService, batch_processing_service,
    BatchProcessingRequest, BatchProcessingResult, MatchingResult
)
from models import CvInput, JdInput, CvMatchResult

logger = logging.getLogger(__name__)

class AgentIntegrationService:
    """
    Service that integrates AI Agent system with existing API endpoints.
    Provides backward compatibility while leveraging new agent architecture.
    """
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.batch_service = batch_processing_service
        
    async def match_cvs_with_agents(self, 
                                  cvs: List[CvInput], 
                                  jd: JdInput,
                                  use_batch_processing: bool = True,
                                  batch_size: int = 20) -> List[CvMatchResult]:
        """
        Match CVs with JD using AI Agent system (optimized version)
        
        Args:
            cvs: List of CV inputs
            jd: Job description input
            use_batch_processing: Whether to use batch processing for optimization
            batch_size: Batch size for processing
            
        Returns:
            List of CvMatchResult compatible with existing API
        """
        if not cvs:
            logger.warning("Empty CV list provided")
            return []
        
        try:
            start_time = time.time()
            
            # For single CV or small batches, use immediate processing
            if len(cvs) <= 3 or not use_batch_processing:
                results = await self._process_immediate(cvs, jd)
            else:
                # Use batch processing for larger datasets
                results = await self._process_batch_optimized(cvs, jd, batch_size)
            
            processing_time = time.time() - start_time
            
            # Calculate LLM call optimization
            original_calls = 1 + 2 * len(cvs)
            optimized_calls = 1 + 2 * ((len(cvs) + batch_size - 1) // batch_size) if use_batch_processing else original_calls
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(f"LLM calls: {original_calls} → {optimized_calls} (saved: {original_calls - optimized_calls})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in agent-based matching: {str(e)}")
            # Fallback to original implementation if needed
            raise

    async def _process_immediate(self, cvs: List[CvInput], jd: JdInput) -> List[CvMatchResult]:
        """Process small batches immediately without queuing"""
        # Extract job_id from JD (assuming it's in the text or we generate one)
        job_id = getattr(jd, 'job_id', f"temp_job_{int(time.time())}")
        cv_ids = [cv.cv_id for cv in cvs]
        
        # Create temporary job data for processing
        job_data = {
            "id": job_id,
            "title": "Job Position",
            "description": jd.text,
            "skillNames": jd.required_skills
        }
        
        # Process using batch service
        batch_result = await self.batch_service.process_job_immediately(
            job_id=job_id,
            cv_ids=cv_ids,
            batch_size=len(cvs)  # Process all at once for small batches
        )
          # Convert to CvMatchResult format
        return self._convert_to_cv_match_results(batch_result.results, cvs)

    async def _process_batch_optimized(self, 
                                     cvs: List[CvInput], 
                                     jd: JdInput, 
                                     batch_size: int) -> List[CvMatchResult]:
        """Process large batches with optimization"""
        job_id = getattr(jd, 'job_id', f"temp_job_{int(time.time())}")
        cv_ids = [cv.cv_id for cv in cvs]
        
        logger.info(f"Starting batch processing for job {job_id}")
        logger.info(f"Input CV IDs: {cv_ids}")
        
        # Submit batch job
        batch_job_id = await self.batch_service.submit_batch_job(
            job_id=job_id,
            cv_ids=cv_ids,
            batch_size=batch_size,
            priority="high"  # API requests have high priority
        )
        
        # Wait for completion with polling
        max_wait_time = 300  # 5 minutes max
        poll_interval = 2    # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = await self.batch_service.get_job_status(batch_job_id)
            if not status:
                raise Exception("Batch job not found")
            
            if status["status"] == "completed":
                result = await self.batch_service.get_job_result(batch_job_id)
                if result:
                    logger.info(f"Batch completed with {len(result.results)} results")
                    for i, match in enumerate(result.results):
                        logger.info(f"Result {i}: cv_id='{match.cv_id}', score={match.overall_score}")
                    return self._convert_to_cv_match_results(result.results, cvs)
                else:
                    raise Exception("Failed to get batch result")
            
            elif status["status"] == "failed":
                error = status.get("error", "Unknown error")
                raise Exception(f"Batch processing failed: {error}")
            
            # Continue polling
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
          # Timeout - try to cancel and fallback
        await self.batch_service.cancel_job(batch_job_id)
        raise Exception("Batch processing timed out")

    def _convert_to_cv_match_results(self, 
                                   matching_results: List[MatchingResult], 
                                   original_cvs: List[CvInput]) -> List[CvMatchResult]:
        """Convert MatchingResult to CvMatchResult for API compatibility"""
        cv_match_results = []
        
        # Create lookup for CV data
        cv_lookup = {cv.cv_id: cv for cv in original_cvs}
        
        logger.info(f"Converting {len(matching_results)} matching results")
        logger.info(f"Original CV IDs: {[cv.cv_id for cv in original_cvs]}")
        
        for i, match in enumerate(matching_results):
            logger.info(f"Processing match {i}: cv_id='{match.cv_id}', score={match.overall_score}")
            
            cv = cv_lookup.get(match.cv_id)
            if not cv:
                logger.warning(f"CV '{match.cv_id}' not found in original list")
                logger.info(f"Available CV IDs: {list(cv_lookup.keys())}")
                continue
            
            # Create enhanced explanation
            explanation = self._create_enhanced_explanation(match)
            
            # Map recommended action
            recommended_action = self._map_recommended_action(match)
            
            cv_match_result = CvMatchResult(
                cv_id=match.cv_id,
                score=match.overall_score,
                explanation=explanation,
                email=cv.email,
                phone=cv.phone,
                recommended_action=recommended_action,
                action_reason=match.action_reason
            )
            
            cv_match_results.append(cv_match_result)
        
        # Sort by score (highest first)
        cv_match_results.sort(key=lambda x: x.score, reverse=True)
        
        return cv_match_results

    def _create_enhanced_explanation(self, match: MatchingResult) -> str:
        """Create enhanced explanation from matching result"""
        explanation_parts = []
        
        # Overall assessment
        if match.overall_score >= 90:
            explanation_parts.append(f"Excellent match ({match.overall_score:.1f}/100)")
        elif match.overall_score >= 80:
            explanation_parts.append(f"Very good match ({match.overall_score:.1f}/100)")
        elif match.overall_score >= 70:
            explanation_parts.append(f"Good match ({match.overall_score:.1f}/100)")
        elif match.overall_score >= 60:
            explanation_parts.append(f"Moderate match ({match.overall_score:.1f}/100)")
        else:
            explanation_parts.append(f"Below average match ({match.overall_score:.1f}/100)")
        
        # Skills analysis
        if match.matched_skills:
            explanation_parts.append(f"Matched skills: {', '.join(match.matched_skills[:5])}")
        
        if match.missing_skills:
            explanation_parts.append(f"Missing skills: {', '.join(match.missing_skills[:3])}")
        
        # Strengths and concerns
        if match.strengths:
            explanation_parts.append(f"Strengths: {', '.join(match.strengths[:2])}")
        
        if match.concerns:
            explanation_parts.append(f"Concerns: {', '.join(match.concerns[:2])}")
        
        # Detailed breakdown if available
        if match.explanation:
            explanation_parts.append(match.explanation)
        
        return ". ".join(explanation_parts)

    def _map_recommended_action(self, match: MatchingResult) -> Optional[str]:
        """Map MatchingResult action to API action format"""
        action_mapping = {
            "send_contact_email": "send_contact_email",
            "request_interview": "send_contact_email",  # Map to existing action
            "save_cv": "save_cv",
            "save_for_later": "save_cv",
            "no_action": None,
            "reject_candidate": None
        }
        
        return action_mapping.get(match.recommended_action, None)

    async def single_cv_match(self, cv: CvInput, jd: JdInput) -> CvMatchResult:
        """Match single CV with JD using agents"""
        results = await self.match_cvs_with_agents([cv], jd, use_batch_processing=False)
        
        if not results:
            raise Exception("Failed to match CV")
        
        return results[0]

    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of the agent integration service"""
        try:
            orchestrator_health = await self.orchestrator.health_check()
            batch_service_health = await self.batch_service.health_check()
            
            return {
                "status": "healthy",
                "orchestrator": orchestrator_health,
                "batch_service": batch_service_health,
                "timestamp": time.time()
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the service"""
        batch_metrics = await self.batch_service.get_service_metrics()
        
        return {
            "agent_integration_service": {
                "status": "active",
                "batch_processing": batch_metrics
            }
        }

# Global service instance
agent_integration_service = AgentIntegrationService()
