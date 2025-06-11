"""
Batch Processing Service - Optimized batch processing for CV-JD matching
Reduces LLM calls from 1+2N to 1+2×ceil(N/batch_size)
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .orchestrator_agent import OrchestratorAgent
from .agent_models import (
    BatchProcessingRequest, BatchProcessingResult, MatchingResult,
    AgentMessage, MessageType
)

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchJob:
    """Represents a batch processing job"""
    job_id: str
    cv_ids: List[str]
    batch_size: int
    priority: str
    status: ProcessingStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[BatchProcessingResult] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    
    @property
    def processing_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

class BatchProcessingService:
    """
    Service for managing batch processing of CV-JD matching operations.
    Optimizes performance by reducing LLM calls through intelligent batching.
    """
    
    def __init__(self, max_concurrent_jobs: int = 3, default_batch_size: int = 20):
        self.orchestrator = OrchestratorAgent()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.default_batch_size = default_batch_size
        
        # Job management
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[str] = []
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        # Performance metrics
        self.total_jobs_processed = 0
        self.total_processing_time = 0.0
        self.total_llm_calls_saved = 0
        
        logger.info(f"Batch Processing Service initialized with max_concurrent_jobs={max_concurrent_jobs}")

    async def submit_batch_job(self, 
                             job_id: str, 
                             cv_ids: List[str],
                             batch_size: Optional[int] = None,
                             priority: str = "normal") -> str:
        """
        Submit a batch processing job
        
        Args:
            job_id: Job ID for the position
            cv_ids: List of CV IDs to process
            batch_size: Batch size for processing (default: service default)
            priority: Priority level (low, normal, high)
            
        Returns:
            Batch job ID for tracking
        """
        if batch_size is None:
            batch_size = self.default_batch_size
            
        batch_job_id = f"batch_{job_id}_{int(time.time())}"
        
        batch_job = BatchJob(
            job_id=job_id,
            cv_ids=cv_ids,
            batch_size=batch_size,
            priority=priority,
            status=ProcessingStatus.PENDING,
            created_at=time.time()
        )
        
        self.active_jobs[batch_job_id] = batch_job
        
        # Add to queue based on priority
        if priority == "high":
            self.job_queue.insert(0, batch_job_id)
        else:
            self.job_queue.append(batch_job_id)
        
        logger.info(f"Batch job {batch_job_id} submitted for job {job_id} with {len(cv_ids)} CVs")
        
        # Start processing if resources available
        asyncio.create_task(self._process_queue())
        
        return batch_job_id

    async def get_job_status(self, batch_job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job"""
        if batch_job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[batch_job_id]
        
        status_info = {
            "batch_job_id": batch_job_id,
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "total_cvs": len(job.cv_ids),
            "batch_size": job.batch_size,
            "priority": job.priority,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "processing_time": job.processing_time,
            "error": job.error
        }
        
        if job.result:
            status_info["result_summary"] = {
                "processed_cvs": job.result.processed_cvs,
                "failed_cvs": job.result.failed_cvs,
                "average_score": job.result.summary.get("average_score"),
                "high_matches": job.result.summary.get("score_distribution", {}).get("excellent", 0)
            }
        
        return status_info

    async def get_job_result(self, batch_job_id: str) -> Optional[BatchProcessingResult]:
        """Get result of a completed batch job"""
        if batch_job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[batch_job_id]
        
        if job.status == ProcessingStatus.COMPLETED:
            return job.result
        
        return None

    async def cancel_job(self, batch_job_id: str) -> bool:
        """Cancel a batch job if it's still pending"""
        if batch_job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[batch_job_id]
        
        if job.status == ProcessingStatus.PENDING:
            job.status = ProcessingStatus.CANCELLED
            if batch_job_id in self.job_queue:
                self.job_queue.remove(batch_job_id)
            logger.info(f"Batch job {batch_job_id} cancelled")
            return True
        
        return False

    async def _process_queue(self):
        """Process jobs from the queue"""
        while self.job_queue:
            # Check if we can process more jobs
            if self.processing_semaphore.locked():
                break
            
            # Get next job from queue
            batch_job_id = self.job_queue.pop(0)
            
            if batch_job_id not in self.active_jobs:
                continue
            
            job = self.active_jobs[batch_job_id]
            
            if job.status != ProcessingStatus.PENDING:
                continue
            
            # Start processing
            asyncio.create_task(self._process_batch_job(batch_job_id))

    async def _process_batch_job(self, batch_job_id: str):
        """Process a single batch job"""
        async with self.processing_semaphore:
            job = self.active_jobs[batch_job_id]
            
            try:
                job.status = ProcessingStatus.PROCESSING
                job.started_at = time.time()
                job.progress = 0
                
                logger.info(f"Starting batch job {batch_job_id} for job {job.job_id}")
                
                # Calculate LLM call optimization
                original_calls = 1 + 2 * len(job.cv_ids)
                optimized_calls = 1 + 2 * ((len(job.cv_ids) + job.batch_size - 1) // job.batch_size)
                calls_saved = original_calls - optimized_calls
                
                logger.info(f"LLM call optimization: {original_calls} → {optimized_calls} (saved: {calls_saved})")
                
                # Process using orchestrator with progress tracking
                result = await self._process_with_progress(job)
                
                job.result = result
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = 100
                
                # Update metrics
                self.total_jobs_processed += 1
                self.total_processing_time += job.processing_time or 0
                self.total_llm_calls_saved += calls_saved
                
                logger.info(f"Batch job {batch_job_id} completed successfully in {job.processing_time:.2f}s")
                
            except Exception as e:
                job.status = ProcessingStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                
                logger.error(f"Batch job {batch_job_id} failed: {str(e)}")
            
            finally:
                # Continue processing queue
                asyncio.create_task(self._process_queue())

    async def _process_with_progress(self, job: BatchJob) -> BatchProcessingResult:
        """Process batch job with progress tracking"""
        # Create progress callback
        total_steps = 3  # JD analysis, CV analysis, Matching
        current_step = 0
        
        def update_progress():
            nonlocal current_step
            current_step += 1
            job.progress = int((current_step / total_steps) * 100)
        
        # Use orchestrator for actual processing
        result = await self.orchestrator.batch_process_job(
            job_id=job.job_id,
            cv_ids=job.cv_ids,
            batch_size=job.batch_size
        )
        
        return result

    async def process_job_immediately(self, 
                                    job_id: str, 
                                    cv_ids: List[str],
                                    batch_size: Optional[int] = None) -> BatchProcessingResult:
        """
        Process a job immediately without queuing (for urgent requests)
        
        Args:
            job_id: Job ID for the position
            cv_ids: List of CV IDs to process
            batch_size: Batch size for processing
            
        Returns:
            BatchProcessingResult
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        
        logger.info(f"Processing job {job_id} immediately with {len(cv_ids)} CVs")
        
        start_time = time.time()
        
        try:
            result = await self.orchestrator.batch_process_job(
                job_id=job_id,
                cv_ids=cv_ids,
                batch_size=batch_size
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Immediate processing completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Immediate processing failed: {str(e)}")
            raise

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        active_count = len([j for j in self.active_jobs.values() if j.status == ProcessingStatus.PROCESSING])
        pending_count = len([j for j in self.active_jobs.values() if j.status == ProcessingStatus.PENDING])
        completed_count = len([j for j in self.active_jobs.values() if j.status == ProcessingStatus.COMPLETED])
        failed_count = len([j for j in self.active_jobs.values() if j.status == ProcessingStatus.FAILED])
        
        avg_processing_time = (
            self.total_processing_time / self.total_jobs_processed 
            if self.total_jobs_processed > 0 else 0
        )
        
        return {
            "service_status": "healthy",
            "active_jobs": active_count,
            "pending_jobs": pending_count,
            "completed_jobs": completed_count,
            "failed_jobs": failed_count,
            "total_jobs_processed": self.total_jobs_processed,
            "average_processing_time": avg_processing_time,
            "total_llm_calls_saved": self.total_llm_calls_saved,
            "queue_length": len(self.job_queue),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "default_batch_size": self.default_batch_size
        }

    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        jobs_to_remove = []
        
        for batch_job_id, job in self.active_jobs.items():
            if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
                if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                    jobs_to_remove.append(batch_job_id)
        
        for batch_job_id in jobs_to_remove:
            del self.active_jobs[batch_job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the batch processing service"""
        try:
            orchestrator_health = await self.orchestrator.health_check()
            service_metrics = await self.get_service_metrics()
            
            # Check if service is overloaded
            pending_jobs = service_metrics["pending_jobs"]
            active_jobs = service_metrics["active_jobs"]
            
            if pending_jobs > 10:
                status = "overloaded"
            elif active_jobs >= self.max_concurrent_jobs:
                status = "busy"
            else:
                status = "healthy"
            
            return {
                "service_status": status,
                "orchestrator_health": orchestrator_health,
                "metrics": service_metrics,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "service_status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Global service instance
batch_processing_service = BatchProcessingService()
