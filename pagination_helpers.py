"""
Enhanced main.py with proper handling of paginated API responses
"""
import httpx
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Add this helper function to main.py to handle paginated job API

async def get_all_recommended_jobs_for_user(cv_id: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Get all recommended jobs for a CV from paginated API
    
    Args:
        cv_id: CV ID to get jobs for
        headers: HTTP headers for authentication
        
    Returns:
        List of all jobs from all pages
    """
    all_jobs = []
    page = 0
    size = 20
    
    async with httpx.AsyncClient() as client:
        while True:
            try:                # Get jobs for current page
                url = f"http://localhost:8080/api/v1/job/recommended/{cv_id}"
                params = {"page": page, "size": size}
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Handle different response structures
                if data.get("statusCode") == 200:
                    jobs_data = data.get("data", {})
                    content = jobs_data.get("content", [])
                    links = jobs_data.get("links", [])
                else:
                    # Fallback for different response structure
                    content = data.get("data", [])
                    links = []
                
                # Add jobs from current page
                all_jobs.extend(content)
                
                # Check if there's a next page
                has_next = any(link.get("rel") == "next" for link in links)
                
                if not has_next or not content:
                    break
                    
                page += 1
                  # Safety limit to prevent infinite loops
                if page > 50:  # Max 1000 jobs (50 pages * 20 per page)
                    logger.warning(f"Reached maximum page limit for CV {cv_id}")
                    break
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # No more pages
                    break
                else:
                    logger.warning(f"HTTP error getting jobs page {page} for CV {cv_id}: {e.response.status_code}")
                    break
            except Exception as e:
                logger.warning(f"Failed to get jobs page {page} for CV {cv_id}: {str(e)}")
                break
    
    return all_jobs
