"""
Database Integration for Job Recommendations
Functions to save recommendations to RecommendJob entity
"""
import httpx
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import date
from job_recommendation_service import JobRecommendationResult
from dotenv import load_dotenv
import os

load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

logger = logging.getLogger(__name__)

class JobRecommendationDB:
    """
    Database integration for job recommendations
    Simplified to only handle batch operations
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8080/api/v1"
        self.headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

    async def save_job_recommendations_batch(self,
                                           recommendations: List[JobRecommendationResult]) -> Dict[str, Any]:
        """
        Save multiple job recommendations to database using batch API
        Automatically clears old data before inserting new recommendations
        
        Args:
            recommendations: List of JobRecommendationResult to save
            
        Returns:
            Batch save results
        """
        try:
            # Convert all recommendations to RecommendJob entity format
            recommend_jobs_data = []
            for recommendation in recommendations:
                # Ensure matched_skills and missing_skills are properly formatted as lists
                matched_skills = recommendation.matched_skills
                if isinstance(matched_skills, str):
                    matched_skills = [skill.strip() for skill in matched_skills.split(",") if skill.strip()]
                elif not isinstance(matched_skills, list):
                    logger.warning(f"Invalid matched_skills type for job {recommendation.job_id}: {type(matched_skills)} - {matched_skills}")
                    matched_skills = []
                
                missing_skills = recommendation.missing_skills
                if isinstance(missing_skills, str):
                    missing_skills = [skill.strip() for skill in missing_skills.split(",") if skill.strip()]
                elif not isinstance(missing_skills, list):
                    logger.warning(f"Invalid missing_skills type for job {recommendation.job_id}: {type(missing_skills)} - {missing_skills}")
                    missing_skills = []
                    logger.debug(f"Job {recommendation.job_id}: matched_skills={matched_skills}, missing_skills={missing_skills}")
                
                recommend_job_data = {
                    "jobId": str(recommendation.job_id),
                    "cvId": str(recommendation.cv_id), 
                    "userId": recommendation.user_id,
                    "overallScore": float(recommendation.overall_score),
                    "matchedSkills": matched_skills,  # Send as array/list
                    "missingSkills": missing_skills,  # Send as array/list
                    "recommendationReason": recommendation.recommendation_reason[:500],  # Limit to 500 chars
                    "createdDate": recommendation.created_date.isoformat()
                }
                recommend_jobs_data.append(recommend_job_data)
            
            if recommend_jobs_data:
                sample_item = recommend_jobs_data[0]
                logger.info(f"Sample recommendation data being sent:")
                logger.info(f"  - jobId: {sample_item['jobId']} (type: {type(sample_item['jobId'])})")
                logger.info(f"  - matchedSkills: {sample_item['matchedSkills']} (type: {type(sample_item['matchedSkills'])})")
                logger.info(f"  - missingSkills: {sample_item['missingSkills']} (type: {type(sample_item['missingSkills'])})")
                logger.info(f"Total recommendations to send: {len(recommend_jobs_data)}")
              
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/recommends/batch"
                
                logger.info(f"Sending POST request to: {url}")
                logger.info(f"Request headers: {self.headers}")
                
                response = await client.post(url, json=recommend_jobs_data, headers=self.headers)
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                
                result = response.json()
                data = result.get("data", {})
                
                return {
                    "success": result.get("success", True),
                    "total_sent": len(recommend_jobs_data),
                    "total_saved": data.get("savedCount", len(recommend_jobs_data)),
                    "failed_count": data.get("failedCount", 0),
                    "message": f"Successfully batch saved {len(recommend_jobs_data)} job recommendations"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error batch saving recommendations: {e.response.status_code} - {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "message": "Failed to batch save job recommendations"
            }
        except Exception as e:
            logger.error(f"Error batch saving recommendations: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to batch save job recommendations"
            }

    async def save_job_recommendations_batch_by_user(self,
                                                   user_recommendations: Dict[str, List[JobRecommendationResult]]) -> Dict[str, Any]:
        """
        Save job recommendations grouped by user using batch API
        Collects all recommendations and saves them in one API call
        
        Args:
            user_recommendations: Dictionary mapping user_id to list of recommendations
            
        Returns:
            Batch save results summary
        """
        # Flatten all recommendations into one list
        all_recommendations = []
        for user_id, recommendations in user_recommendations.items():
            all_recommendations.extend(recommendations)
        
        if not all_recommendations:
            return {
                "success": True,
                "saved_recommendations": 0,
                "message": "No recommendations to save"
            }
        
        # Use the single batch save method
        result = await self.save_job_recommendations_batch(all_recommendations)
        
        return {
            "success": result.get("success", False),
            "saved_recommendations": result.get("total_saved", 0),
            "failed_recommendations": result.get("failed_count", 0),
            "total_users": len(user_recommendations),
            "total_recommendations": len(all_recommendations),
            "message": result.get("message", "Batch save completed")
        }

# Global service instance
job_recommendation_db = JobRecommendationDB()
