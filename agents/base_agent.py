"""
Base Agent Interface - Abstract class for all AI Agents
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import time
from datetime import datetime
import asyncio

from .agent_models import AgentMessage, AgentPerformanceMetrics, MessageType

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base abstract class for all AI Agents
    Provides common functionality and interface
    """
    
    def __init__(self, agent_id: str, model_name: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.metrics = AgentPerformanceMetrics(agent_id=agent_id)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response
        
        Args:
            message: AgentMessage containing request data
            
        Returns:
            AgentMessage containing response data
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent
        
        Returns:
            System prompt string
        """
        pass
    
    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Call LLM with error handling and metrics tracking
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters for LLM call
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from openai_service import ai_agent_chat
              # Call LLM (ai_agent_chat doesn't accept these parameters directly)
            result = await ai_agent_chat(messages, use_functions=False)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.total_llm_calls += 1
            self.metrics.total_requests += 1
            
            if result.get("success", False):
                self.metrics.successful_requests += 1
                return result
            else:
                self.metrics.failed_requests += 1
                raise Exception(f"LLM call failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.failed_requests += 1
            self.logger.error(f"LLM call failed for {self.agent_id}: {str(e)}")
            raise
        finally:
            # Update average processing time
            total_time = self.metrics.average_processing_time * (self.metrics.total_requests - 1)
            self.metrics.average_processing_time = (total_time + processing_time) / self.metrics.total_requests
    
    def _create_response_message(self, 
                               message_type: MessageType,
                               data: Dict[str, Any],
                               confidence: float = 1.0,
                               metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """
        Create a standardized response message
        
        Args:
            message_type: Type of response message
            data: Response data
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            
        Returns:
            AgentMessage response
        """
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "processing_agent": self.agent_id,
            "model_used": self.model_name,
            "timestamp": datetime.now().isoformat()
        })
        
        return AgentMessage(
            agent_id=self.agent_id,
            message_type=message_type,
            data=data,
            confidence=confidence,
            metadata=metadata
        )
    
    def _create_error_message(self, error: str, original_message: AgentMessage) -> AgentMessage:
        """
        Create an error response message
        
        Args:
            error: Error description
            original_message: Original request message
            
        Returns:
            AgentMessage with error
        """
        return self._create_response_message(
            message_type=MessageType.ERROR,
            data={
                "error": error,
                "original_request": original_message.dict()
            },
            confidence=0.0,
            metadata={"error_occurred": True}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for this agent
        
        Returns:
            Health status information
        """
        try:
            # Simple test message
            test_messages = [
                {"role": "system", "content": "You are a test agent."},
                {"role": "user", "content": "Respond with 'OK' if you are working."}
            ]
            
            result = await self._call_llm(test_messages, max_tokens=10)
            
            return {
                "agent_id": self.agent_id,
                "status": "healthy",
                "model": self.model_name,
                "metrics": self.metrics.dict(),
                "test_response": result.get("message", "")
            }
            
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.metrics.dict()
            }
    
    def get_metrics(self) -> AgentPerformanceMetrics:
        """
        Get current performance metrics
        
        Returns:
            AgentPerformanceMetrics object
        """
        return self.metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = AgentPerformanceMetrics(agent_id=self.agent_id)
