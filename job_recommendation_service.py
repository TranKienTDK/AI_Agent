"""
Job Recommendation Service - Integration service for CV-to-Jobs matching
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import date
import httpx

from agents import CVAnalyzerAgent, JDAnalyzerAgent
from agents.job_recommendation_agent import JobRecommendationAgent, JobRecommendationResult
from agents.agent_models import AgentMessage, MessageType, CVAnalysisResult, JDAnalysisResult
from models import CvInput, JdInput

logger = logging.getLogger(__name__)

class JobRecommendationService:
    """
    Service for recommending jobs to candidates using AI Agent system
    """
    
    def __init__(self):
        self.cv_analyzer = CVAnalyzerAgent()
        self.jd_analyzer = JDAnalyzerAgent()
        self.job_recommendation_agent = JobRecommendationAgent()
        
    async def recommend_jobs_for_candidate(self,
                                         cv_data: Dict[str, Any],
                                         job_list: List[Dict[str, Any]],
                                         user_id: str,
                                         cv_id: str = None,
                                         min_score: float = 50.0) -> List[JobRecommendationResult]:
        """
        Recommend jobs for a candidate
        
        Args:
            cv_data: Raw CV data from database
            job_list: List of pre-filtered job data from database
            user_id: User ID for the candidate
            cv_id: CV ID for the candidate
            min_score: Minimum score threshold for recommendations
            
        Returns:
            List of JobRecommendationResult objects
        """
        try:
            start_time = time.time()
            
            logger.info(f"Starting job recommendation for user {user_id} with {len(job_list)} jobs")
            
            # Step 1: Analyze CV
            cv_analysis = await self._analyze_cv(cv_data)
            
            # Step 2: Analyze Jobs in batch
            job_analyses = await self._analyze_jobs_batch(job_list)
              
            # Step 3: Generate recommendations
            recommendations = await self._generate_recommendations(cv_analysis, job_analyses, user_id, cv_id)
            
            # Step 4: Filter by minimum score
            filtered_recommendations = [
                rec for rec in recommendations 
                if rec.overall_score >= min_score
            ]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Job recommendation completed in {processing_time:.2f}s")
            logger.info(f"Found {len(filtered_recommendations)} suitable jobs out of {len(job_list)}")
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error in job recommendation: {str(e)}")
            raise

    async def _analyze_cv(self, cv_data: Dict[str, Any]) -> CVAnalysisResult:
        """Analyze CV using CV Analyzer Agent"""
        message = AgentMessage(
            agent_id="job_recommendation_service",
            message_type=MessageType.ANALYSIS_REQUEST,
            data={"cv_data": cv_data}
        )
        
        result = await self.cv_analyzer.process(message)
        
        if result.message_type == MessageType.ERROR:
            raise Exception(f"CV analysis failed: {result.data}")
        
        cv_analyses = result.data["cv_analyses"]
        return CVAnalysisResult(**cv_analyses[0])

    async def _analyze_jobs_batch(self, job_list: List[Dict[str, Any]]) -> List[JDAnalysisResult]:
        """Analyze jobs in batch using JD Analyzer Agent"""
        job_analyses = []
        
        # Process jobs in chunks for efficiency
        chunk_size = 10
        for i in range(0, len(job_list), chunk_size):
            chunk = job_list[i:i + chunk_size]
            chunk_analyses = await self._analyze_job_chunk(chunk)
            job_analyses.extend(chunk_analyses)
            
            # Small delay between chunks
            if i + chunk_size < len(job_list):
                await asyncio.sleep(0.5)
        
        return job_analyses

    async def _analyze_job_chunk(self, job_chunk: List[Dict[str, Any]]) -> List[JDAnalysisResult]:
        """Analyze a chunk of jobs"""
        tasks = [self._analyze_single_job(job_data) for job_data in job_chunk]
        return await asyncio.gather(*tasks)

    async def _analyze_single_job(self, job_data: Dict[str, Any]) -> JDAnalysisResult:
        """Analyze a single job"""
        message = AgentMessage(
            agent_id="job_recommendation_service",
            message_type=MessageType.ANALYSIS_REQUEST,
            data={"job_data": job_data}
        )
        
        result = await self.jd_analyzer.process(message)
        
        if result.message_type == MessageType.ERROR:
            # Log warning but continue with other jobs
            logger.warning(f"Job analysis failed for job {job_data.get('id', 'unknown')}: {result.data}")
            # Return a basic analysis
            return JDAnalysisResult(
                job_id=job_data.get("id", ""),
                required_skills=job_data.get("skillNames", []),
                confidence=0.5
            )
        
        return JDAnalysisResult(**result.data["jd_analysis"])

    async def _generate_recommendations(self,
                                      cv_analysis: CVAnalysisResult,
                                      job_analyses: List[JDAnalysisResult],
                                      user_id: str,
                                      cv_id: str = None) -> List[JobRecommendationResult]:
        """Generate job recommendations"""
        message = AgentMessage(
            agent_id="job_recommendation_service",
            message_type=MessageType.ANALYSIS_REQUEST,
            data={
                "cv_analysis": cv_analysis.dict(),
                "job_analyses": [job.dict() for job in job_analyses],
                "user_id": user_id,
                "cv_id": cv_id or user_id  # Use cv_id if available, fallback to user_id
            }
        )
        
        result = await self.job_recommendation_agent.process(message)
        
        if result.message_type == MessageType.ERROR:
            raise Exception(f"Job recommendation failed: {result.data}")
        
        recommendations_data = result.data["job_recommendations"]
        return [
            JobRecommendationResult(
                cv_id=rec["cv_id"],
                job_id=rec["job_id"],
                user_id=rec["user_id"],
                overall_score=rec["overall_score"],
                matched_skills=rec["matched_skills"],
                missing_skills=rec["missing_skills"],
                recommendation_reason=rec["recommendation_reason"]
            )
            for rec in recommendations_data
        ]
        
    async def batch_recommend_for_users(self,
                                      user_cv_data: List[Dict[str, Any]],
                                      user_jobs_map: Dict[str, List[Dict[str, Any]]],
                                      min_score: float = 50.0) -> Dict[str, List[JobRecommendationResult]]:
        """
        Batch recommend jobs for multiple users with their specific job lists
        
        Args:
            user_cv_data: List of user CV data with user_id
            user_jobs_map: Dictionary mapping user_id to their job list
            min_score: Minimum score threshold
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        results = {}
        
        for user_cv in user_cv_data:
            user_id = user_cv.get("user_id")
            cv_data = user_cv.get("cv_data")
            user_jobs = user_jobs_map.get(user_id, [])
            
            if not user_id or not cv_data or not user_jobs:
                results[user_id] = []
                continue
                
            try:
                recommendations = await self.recommend_jobs_for_candidate(
                    cv_data=cv_data,
                    job_list=user_jobs,
                    user_id=user_id,
                    min_score=min_score
                )
                results[user_id] = recommendations
                
            except Exception as e:
                logger.error(f"Failed to recommend jobs for user {user_id}: {str(e)}")
                results[user_id] = []
        
        return results

    async def _get_all_jobs_from_paginated_api(self, user_id: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Get all jobs from paginated API for a user
        
        Args:
            user_id: User ID to get recommended jobs for
            headers: HTTP headers for authentication
            
        Returns:
            List of all jobs from all pages
        """
        all_jobs = []
        page = 0
        size = 20
        
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    # Get jobs for current page
                    url = f"http://localhost:8080/api/v1/job/recommended/{user_id}"
                    params = {"page": page, "size": size}
                    
                    response = await client.get(url, headers=headers, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    jobs_data = data.get("data", {})
                    content = jobs_data.get("content", [])
                    links = jobs_data.get("links", [])
                    
                    # Add jobs from current page
                    all_jobs.extend(content)
                    
                    # Check if there's a next page
                    has_next = any(link.get("rel") == "next" for link in links)
                    
                    if not has_next or not content:
                        break
                        
                    page += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to get jobs page {page} for user {user_id}: {str(e)}")
                    break
        
        return all_jobs

# Global service instance
job_recommendation_service = JobRecommendationService()
