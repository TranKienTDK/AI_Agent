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

class CvMatchResult(BaseModel):
    cv_id: str
    score: float
    explanation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None