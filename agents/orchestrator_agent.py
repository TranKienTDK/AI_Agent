"""
Orchestrator Agent - Coordinator agent for managing workflow between specialized agents
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .agent_models import (
    AgentMessage, MessageType, BatchProcessingRequest, BatchProcessingResult,
    JDAnalysisResult, CVAnalysisResult, MatchingResult
)
from .jd_analyzer_agent import JDAnalyzerAgent
from .cv_analyzer_agent import CVAnalyzerAgent
from .matching_agent import MatchingAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent coordinates the workflow between JD Analyzer, CV Analyzer, and Matching agents.
    Manages batch processing to optimize LLM calls from 1+2N to 1+2×ceil(N/batch_size).
    """
    
    def __init__(self):
        super().__init__(agent_id="orchestrator")
        self.jd_analyzer = JDAnalyzerAgent()
        self.cv_analyzer = CVAnalyzerAgent()
        self.matching_agent = MatchingAgent()
        
    def get_system_prompt(self) -> str:
        return """
You are the Orchestrator Agent, responsible for coordinating the CV-JD matching workflow.

Your responsibilities:
1. **Workflow Coordination**: Manage the sequence of JD analysis → CV analysis → Matching
2. **Batch Processing**: Optimize LLM calls by batching CV analyses
3. **Error Handling**: Gracefully handle failures and provide fallback mechanisms
4. **Performance Monitoring**: Track processing time and success rates
5. **Result Aggregation**: Combine results from all agents into comprehensive reports

You coordinate between:
- JD Analyzer Agent: Analyzes job descriptions
- CV Analyzer Agent: Analyzes CVs in batches
- Matching Agent: Performs CV-JD matching with batch processing

Your goal is to reduce LLM calls from 1+2N to 1+2×ceil(N/batch_size) through intelligent batching.
"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process orchestration request - can handle single match or batch processing
        
        Args:
            message: AgentMessage containing job_id and cv_ids for processing
            
        Returns:
            AgentMessage with orchestrated results
        """
        try:
            if message.message_type == MessageType.BATCH_PROCESSING_REQUEST:
                return await self._process_batch_request(message)
            elif message.message_type == MessageType.SINGLE_MATCH_REQUEST:
                return await self._process_single_match(message)
            else:
                raise ValueError(f"Unsupported message type: {message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            return self._create_error_message(str(e), message)

    async def _process_batch_request(self, message: AgentMessage) -> AgentMessage:
        """Process batch matching request with optimized LLM calls"""
        batch_request = BatchProcessingRequest(**message.data)
        start_time = time.time()
        
        self.logger.info(f"Starting batch processing for job {batch_request.job_id} with {len(batch_request.cv_ids)} CVs")
        
        try:
            # Step 1: Analyze JD (1 LLM call)
            jd_analysis = await self._analyze_job_description(batch_request.job_id)
            
            # Step 2: Analyze CVs in batches (ceil(N/batch_size) LLM calls)
            cv_analyses = await self._analyze_cvs_batch(batch_request.cv_ids, batch_request.batch_size)
            
            # Step 3: Perform matching in batches (ceil(N/batch_size) LLM calls)
            matching_results = await self._match_candidates_batch(jd_analysis, cv_analyses, batch_request.batch_size)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create batch result
            batch_result = BatchProcessingResult(
                job_id=batch_request.job_id,
                total_cvs=len(batch_request.cv_ids),
                processed_cvs=len(matching_results),
                failed_cvs=len(batch_request.cv_ids) - len(matching_results),
                processing_time=processing_time,
                results=matching_results,
                errors=[],
                summary=self._create_batch_summary(matching_results)
            )
            
            self.logger.info(f"Batch processing completed in {processing_time:.2f}s")
            
            return self._create_response_message(
                message_type=MessageType.BATCH_PROCESSING_RESULT,
                data={"batch_result": batch_result.dict()},
                confidence=0.9,
                metadata={
                    "processing_time": processing_time,
                    "total_llm_calls": 1 + 2 * ((len(batch_request.cv_ids) + batch_request.batch_size - 1) // batch_request.batch_size),
                    "optimization_ratio": f"{1 + 2 * len(batch_request.cv_ids)} -> {1 + 2 * ((len(batch_request.cv_ids) + batch_request.batch_size - 1) // batch_request.batch_size)}"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return self._create_error_message(str(e), message)

    async def _process_single_match(self, message: AgentMessage) -> AgentMessage:
        """Process single CV-JD match"""
        job_id = message.data.get("job_id")
        cv_id = message.data.get("cv_id")
        
        if not job_id or not cv_id:
            raise ValueError("Missing job_id or cv_id")
        
        self.logger.info(f"Processing single match for CV {cv_id} with job {job_id}")
        
        try:
            # Analyze JD
            jd_analysis = await self._analyze_job_description(job_id)
            
            # Analyze CV
            cv_analyses = await self._analyze_cvs_batch([cv_id], batch_size=1)
            
            if not cv_analyses:
                raise Exception("Failed to analyze CV")
            
            # Perform matching
            matching_results = await self._match_candidates_batch(jd_analysis, cv_analyses, batch_size=1)
            
            if not matching_results:
                raise Exception("Failed to match candidate")
            
            return self._create_response_message(
                message_type=MessageType.MATCHING_RESULT,
                data={"matching_result": matching_results[0].dict()},
                confidence=matching_results[0].confidence,
                metadata={
                    "job_id": job_id,
                    "cv_id": cv_id,
                    "processing_type": "single_match"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in single match: {str(e)}")
            return self._create_error_message(str(e), message)

    async def _analyze_job_description(self, job_id: str) -> JDAnalysisResult:
        """Analyze job description using JD Analyzer Agent"""
        # Get job data first
        from function_tools import ai_tools
        
        job_result = await ai_tools.get_job_data(job_id)
        if not job_result.get("success", False):
            raise Exception(f"Failed to get job data: {job_result.get('error', 'Unknown error')}")
        
        job_data = job_result["job_info"]
        
        message = AgentMessage(
            agent_id=self.agent_id,
            message_type=MessageType.ANALYSIS_REQUEST,
            data={"job_data": job_data}
        )
        
        result = await self.jd_analyzer.process(message)
        
        if result.message_type == MessageType.ERROR:
            raise Exception(f"JD analysis failed: {result.data}")
        
        return JDAnalysisResult(**result.data["jd_analysis"])

    async def _analyze_cvs_batch(self, cv_ids: List[str], batch_size: int) -> List[CVAnalysisResult]:
        """Analyze CVs in batches using CV Analyzer Agent"""
        all_results = []
          # Process CVs in chunks
        for i in range(0, len(cv_ids), batch_size):
            chunk = cv_ids[i:i + batch_size]
            
            message = AgentMessage(
                agent_id=self.agent_id,
                message_type=MessageType.ANALYSIS_REQUEST,
                data={"cv_ids": chunk}
            )
            
            result = await self.cv_analyzer.process(message)
            
            if result.message_type == MessageType.ANALYSIS_RESULT:
                cv_analyses_data = result.data["cv_analyses"]
                chunk_results = [CVAnalysisResult(**cv_data) for cv_data in cv_analyses_data]
                all_results.extend(chunk_results)
            else:
                self.logger.error(f"CV analysis failed for chunk {chunk}: {result.data}")
                # Continue with other chunks
        
        return all_results

    async def _match_candidates_batch(self, 
                                    jd_analysis: JDAnalysisResult, 
                                    cv_analyses: List[CVAnalysisResult],
                                    batch_size: int) -> List[MatchingResult]:
        """Perform matching in batches using Matching Agent"""
        all_results = []
        
        # Process matches in chunks
        for i in range(0, len(cv_analyses), batch_size):
            chunk = cv_analyses[i:i + batch_size]
            
            message = AgentMessage(
                agent_id=self.agent_id,
                message_type=MessageType.MATCHING_REQUEST,
                data={
                    "jd_analysis": jd_analysis.dict(),
                    "cv_analyses": [cv.dict() for cv in chunk]
                }
            )
            
            result = await self.matching_agent.process(message)
            
            if result.message_type == MessageType.MATCHING_RESULT:
                matching_results_data = result.data["matching_results"]
                chunk_results = [MatchingResult(**match_data) for match_data in matching_results_data]
                all_results.extend(chunk_results)
            else:
                self.logger.error(f"Matching failed for chunk: {result.data}")
                # Continue with other chunks
        
        return all_results

    def _create_batch_summary(self, results: List[MatchingResult]) -> Dict[str, Any]:
        """Create summary statistics for batch processing results"""
        if not results:
            return {}
        
        scores = [r.overall_score for r in results]
        actions = [r.recommended_action for r in results if r.recommended_action]
        
        summary = {
            "total_matches": len(results),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "high_confidence_matches": len([r for r in results if r.confidence > 0.8]),
            "recommended_actions": {
                "send_contact_email": len([a for a in actions if a == "send_contact_email"]),
                "save_cv": len([a for a in actions if a == "save_cv"]),
                "request_interview": len([a for a in actions if a == "request_interview"]),
                "no_action": len([r for r in results if not r.recommended_action])
            },
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 90]),
                "good": len([s for s in scores if 80 <= s < 90]),
                "average": len([s for s in scores if 60 <= s < 80]),
                "below_average": len([s for s in scores if s < 60])
            }
        }
        
        return summary

    async def batch_process_job(self, job_id: str, cv_ids: List[str], batch_size: int = 20) -> BatchProcessingResult:
        """
        Convenience method for batch processing a job
        
        Args:
            job_id: Job ID to process
            cv_ids: List of CV IDs to match
            batch_size: Batch size for processing
            
        Returns:
            BatchProcessingResult
        """
        request = BatchProcessingRequest(
            job_id=job_id,
            cv_ids=cv_ids,
            batch_size=batch_size
        )
        
        message = AgentMessage(
            agent_id="external",
            message_type=MessageType.BATCH_PROCESSING_REQUEST,
            data=request.dict()
        )
        
        result = await self.process(message)
        
        if result.message_type == MessageType.ERROR:
            raise Exception(f"Batch processing failed: {result.data}")
        
        return BatchProcessingResult(**result.data["batch_result"])

    async def single_match(self, job_id: str, cv_id: str) -> MatchingResult:
        """
        Convenience method for single CV-JD matching
        
        Args:
            job_id: Job ID
            cv_id: CV ID to match
            
        Returns:
            MatchingResult
        """
        message = AgentMessage(
            agent_id="external",
            message_type=MessageType.SINGLE_MATCH_REQUEST,
            data={"job_id": job_id, "cv_id": cv_id}
        )
        
        result = await self.process(message)
        
        if result.message_type == MessageType.ERROR:
            raise Exception(f"Single match failed: {result.data}")
        
        return MatchingResult(**result.data["matching_result"])

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all sub-agents"""
        health_status = await super().health_check()
        
        # Check sub-agents
        sub_agents = {
            "jd_analyzer": self.jd_analyzer,
            "cv_analyzer": self.cv_analyzer,
            "matching_agent": self.matching_agent
        }
        
        for name, agent in sub_agents.items():
            try:
                agent_health = await agent.health_check()
                health_status[f"sub_agent_{name}"] = agent_health
            except Exception as e:
                health_status[f"sub_agent_{name}"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Overall health based on sub-agents
        all_healthy = all(
            status.get("status") == "healthy" 
            for key, status in health_status.items() 
            if key.startswith("sub_agent_")
        )
        
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
