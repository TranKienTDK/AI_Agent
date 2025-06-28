"""
Scheduler Service for Job Recommendations
Handles scheduled job to run recommendation processing at midnight
"""
import asyncio
import logging
import schedule
import time
from datetime import datetime
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

logger = logging.getLogger(__name__)

class RecommendationScheduler:
    """
    Scheduler for running job recommendations at midnight
    """
    
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
          async def run_daily_recommendation_job(self):
        """
        Main scheduled job that runs at midnight
        1. Clear all existing recommendations
        2. Process all CV-Job matches using the main API
        """
        start_time = datetime.now()
        logger.info(f"Starting daily recommendation job at {start_time}")
        
        try:
            # Step 1: Clear all existing recommendations
            await self._clear_all_recommendations()
            
            # Step 2: Call the main processing API to handle all CV-Job matching
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8000/recommend-jobs/process-all"
                params = {"min_score": 50.0}
                response = await client.post(url, params=params)
                response.raise_for_status()
                
                result = response.json()
                
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Daily recommendation job completed in {duration}")
            logger.info(f"Processed {result['processing_summary']['processed_users']} users")
            logger.info(f"Generated {result['processing_summary']['total_recommendations_generated']} recommendations")
            
            return {
                "success": True,
                "job_result": result,
                "duration_seconds": duration.total_seconds(),
                "completed_at": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Daily recommendation job failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
    
    async def _clear_all_recommendations(self):
        """Clear all existing recommendations from database"""
        try:
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8080/api/v1/recommends/clear-all"
                response = await client.delete(url, headers=self.headers)
                response.raise_for_status()
                logger.info("Successfully cleared all existing recommendations")
        except Exception as e:            logger.error(f"Failed to clear existing recommendations: {str(e)}")
            raise

# Global scheduler instance
recommendation_scheduler = RecommendationScheduler()

def schedule_daily_job():
    """
    Schedule the daily recommendation job to run at midnight
    """
    # Schedule job to run every day at midnight
    schedule.every().day.at("00:00").do(
        lambda: asyncio.create_task(recommendation_scheduler.run_daily_recommendation_job())
    )
    
    logger.info("Scheduled daily recommendation job to run at midnight")

def run_scheduler():
    """
    Run the scheduler in a loop
    This should be called to start the scheduling service
    """
    schedule_daily_job()
    
    logger.info("Starting recommendation scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Manual trigger for testing
async def trigger_manual_recommendation_job():
    """
    Manually trigger the recommendation job for testing
    """
    logger.info("Manually triggering recommendation job...")
    return await recommendation_scheduler.run_daily_recommendation_job()
