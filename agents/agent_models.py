"""
Agent models for AI Agent communication and data structures
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    """Types of messages between agents"""
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESULT = "analysis_result"
    BATCH_ANALYSIS_REQUEST = "batch_analysis_request"
    BATCH_ANALYSIS_RESULT = "batch_analysis_result"
    MATCHING_REQUEST = "matching_request"
    MATCHING_RESULT = "matching_result"
    BATCH_PROCESSING_REQUEST = "batch_processing_request"
    BATCH_PROCESSING_RESULT = "batch_processing_result"
    SINGLE_MATCH_REQUEST = "single_match_request"
    ERROR = "error"
    STATUS = "status"

class AgentMessage(BaseModel):
    """Standard message format for agent communication"""
    agent_id: str
    message_type: MessageType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    confidence: float = 1.0
    timestamp: datetime = None
    
    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = datetime.now()
        super().__init__(**data)

class JDAnalysisResult(BaseModel):
    """Result from JD Analyzer Agent"""
    job_id: str
    required_skills: List[str] = []
    nice_to_have_skills: List[str] = []
    experience_level: str = "mid"  # junior, mid, senior
    experience_years: Optional[int] = None
    soft_skills: List[str] = []
    industry_context: str = ""
    urgency_level: str = "medium"  # low, medium, high
    cultural_requirements: List[str] = []
    education_requirements: List[str] = []
    certification_requirements: List[str] = []
    salary_range: Optional[Dict[str, int]] = None
    work_arrangement: str = "hybrid"  # remote, hybrid, onsite
    confidence: float = 1.0
    raw_analysis: str = ""

class CVAnalysisResult(BaseModel):
    """Result from CV Analyzer Agent"""
    cv_id: str
    skills: List[str] = []
    skill_levels: Dict[str, str] = {}  # skill -> level mapping
    experience_years: Optional[float] = None
    career_level: str = "mid"  # junior, mid, senior
    strengths: List[str] = []
    weaknesses: List[str] = []
    red_flags: List[str] = []
    unique_selling_points: List[str] = []
    career_progression: str = ""
    stability_score: float = 0.5  # 0-1 scale
    growth_potential: str = "medium"  # low, medium, high
    education_background: List[str] = []
    certifications: List[str] = []
    projects: List[Dict[str, str]] = []
    languages: List[Dict[str, str]] = []
    expected_salary: Optional[int] = None
    work_preference: str = "hybrid"
    confidence: float = 1.0
    raw_analysis: str = ""

class MatchingResult(BaseModel):
    """Result from Matching Agent"""
    cv_id: str
    job_id: str
    overall_score: float  # 0-100
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    cultural_fit_score: float
    growth_potential_score: float
    
    # Detailed analysis
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    transferable_skills: List[str] = []
    skill_gaps: List[str] = []
    
    strengths: List[str] = []
    concerns: List[str] = []
    recommendations: List[str] = []
    
    # Actions
    recommended_action: Optional[str] = None
    action_reason: str = ""
    priority_level: str = "medium"
    
    # Meta
    confidence: float = 1.0
    explanation: str = ""
    detailed_breakdown: Dict[str, Any] = {}

class BatchProcessingRequest(BaseModel):
    """Request for batch processing"""
    job_id: str
    cv_ids: List[str]
    batch_size: int = 20
    priority: str = "normal"  # low, normal, high
    metadata: Dict[str, Any] = {}

class BatchProcessingResult(BaseModel):
    """Result from batch processing"""
    job_id: str
    total_cvs: int
    processed_cvs: int
    failed_cvs: int
    processing_time: float
    results: List[MatchingResult] = []
    errors: List[Dict[str, str]] = []
    summary: Dict[str, Any] = {}

class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for agents"""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time: float = 0.0
    total_llm_calls: int = 0
    total_cost: float = 0.0
    confidence_scores: List[float] = []
    last_updated: datetime = None
    
    def __init__(self, **data):
        if data.get('last_updated') is None:
            data['last_updated'] = datetime.now()
        super().__init__(**data)
