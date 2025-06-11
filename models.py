from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class ProjectInput(BaseModel):
    project: str
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class LanguageInput(BaseModel):
    language: str
    level: Optional[str] = None

class CvInput(BaseModel):
    cv_id: str
    skills: Optional[List[str]] = None
    experience: Optional[str] = None
    education: Optional[str] = None
    certifications: Optional[List[str]] = None
    projects: Optional[List[ProjectInput]] = None
    languages: Optional[List[LanguageInput]] = None
    text: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class JdInput(BaseModel):
    required_skills: Optional[List[str]] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    required_certifications: Optional[List[str]] = None
    text: Optional[str] = None
    job_id: Optional[str] = None  # Add job_id field for AI Agent processing

class CvMatchResult(BaseModel):
    cv_id: str
    score: float
    explanation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    recommended_action: Optional[str] = None
    action_reason: Optional[str] = None

class ActionRecommendation(BaseModel):
    action_type: str  # "send_contact_email", "save_cv"
    priority: str  # "high", "medium", "low"
    reason: str
    suggested_next_steps: Optional[List[str]] = None

class AIAgentRecommendation(BaseModel):
    cv_id: str
    job_id: str
    match_score: float
    match_explanation: str
    recommended_actions: List[ActionRecommendation]
    hr_notes: Optional[str] = None