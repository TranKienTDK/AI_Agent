"""
JD Analyzer Agent - Specialized agent for analyzing Job Descriptions
"""
import json
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent
from .agent_models import AgentMessage, MessageType, JDAnalysisResult

logger = logging.getLogger(__name__)

class JDAnalyzerAgent(BaseAgent):
    """
    Specialized AI Agent for analyzing Job Descriptions
    
    Responsibilities:
    - Extract structured information from JD text
    - Classify skills by priority (must-have vs nice-to-have)
    - Identify experience level and requirements
    - Determine cultural fit and soft skill requirements
    - Assess urgency and context
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(agent_id="jd_analyzer", model_name=model_name)
        
    def get_system_prompt(self) -> str:
        return """
You are a specialized HR AI Agent expert in analyzing Job Descriptions (JDs). Your role is to extract structured, actionable information from job postings to enable precise candidate matching.

Your analysis capabilities include:

1. **Skill Classification & Prioritization**:
   - Identify technical skills with precise names (e.g., "React.js", "Python", "AWS")
   - Classify skills as "must-have" vs "nice-to-have" based on language cues
   - Recognize skill variations and synonyms (e.g., "JavaScript" = "JS")

2. **Experience Analysis**:
   - Extract years of experience required
   - Determine seniority level: junior (0-2 years), mid (2-5 years), senior (5+ years)
   - Identify specific experience domains (e.g., "e-commerce", "fintech", "startup")

3. **Context Understanding**:
   - Industry and company size indicators
   - Work arrangement preferences (remote/hybrid/onsite)
   - Team structure and reporting relationships
   - Project types and technologies used

4. **Soft Skills & Culture**:
   - Leadership requirements
   - Communication and collaboration needs
   - Cultural fit indicators
   - Personality traits valued

5. **Requirements Hierarchy**:
   - Education requirements (required vs preferred)
   - Certifications and licenses
   - Language requirements with proficiency levels

6. **Urgency Assessment**:
   - Timeline indicators ("urgent", "immediate start", "ASAP")
   - Market competitiveness signals
   - Budget and salary indicators

**Analysis Framework**:
- Use explicit language cues: "required", "must have", "essential" → must-have
- Use preference language: "preferred", "nice to have", "plus" → nice-to-have  
- Consider industry standards for experience mapping
- Detect hidden requirements through context clues

**Output Requirements**:
- Return valid JSON with structured analysis
- Include confidence scores for uncertain classifications
- Provide reasoning for key decisions
- Flag any ambiguous or missing information

Always be thorough but concise, focusing on actionable insights for candidate matching.
"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process JD analysis request
        
        Args:
            message: AgentMessage containing JD data
            
        Returns:
            AgentMessage with JD analysis results
        """
        try:
            self.logger.info(f"Processing JD analysis request for job: {message.data.get('job_id', 'unknown')}")
            
            # Extract job data
            job_data = message.data.get("job_data", {})
            if not job_data:
                raise ValueError("No job_data provided in message")
            
            # Perform analysis
            analysis_result = await self._analyze_job_description(job_data)
            
            # Create response
            return self._create_response_message(
                message_type=MessageType.ANALYSIS_RESULT,
                data={"jd_analysis": analysis_result.dict()},
                confidence=analysis_result.confidence,
                metadata={
                    "job_id": analysis_result.job_id,
                    "analysis_type": "job_description"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing JD analysis: {str(e)}")
            return self._create_error_message(str(e), message)
    
    async def _analyze_job_description(self, job_data: Dict[str, Any]) -> JDAnalysisResult:
        """
        Analyze job description using LLM
        
        Args:
            job_data: Raw job data from API
            
        Returns:
            JDAnalysisResult with structured analysis
        """
        # Prepare analysis prompt
        job_id = job_data.get("id", "")
        title = job_data.get("title", "")
        description = job_data.get("description", "")
        skill_names = job_data.get("skillNames", [])
        
        # Create comprehensive prompt
        analysis_prompt = f"""
Analyze this Job Description and return a comprehensive structured analysis in JSON format.

**Job Information:**
- Title: {title}
- Description: {description}
- Listed Skills: {skill_names}

**Required JSON Output Format:**
{{
    "required_skills": ["skill1", "skill2", ...],
    "nice_to_have_skills": ["skill3", "skill4", ...],
    "experience_level": "junior|mid|senior",
    "experience_years": number or null,
    "soft_skills": ["communication", "leadership", ...],
    "industry_context": "description of industry/domain",
    "urgency_level": "low|medium|high",
    "cultural_requirements": ["team player", "self-starter", ...],
    "education_requirements": ["Bachelor's degree", ...],
    "certification_requirements": ["AWS certification", ...],
    "work_arrangement": "remote|hybrid|onsite",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of key decisions"
}}

**Analysis Guidelines:**
1. Classify skills based on language cues in the description
2. Map experience requirements to our levels (junior: 0-2, mid: 2-5, senior: 5+)
3. Extract soft skills and cultural indicators
4. Assess urgency from timeline language
5. Identify education and certification needs
6. Determine work arrangement preferences

Return ONLY the JSON object, no additional text.
"""

        # Call LLM for analysis
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": analysis_prompt}
        ]
        
        result = await self._call_llm(messages, max_tokens=1500)
        
        if not result.get("success", False):
            raise Exception(f"LLM analysis failed: {result.get('error', 'Unknown error')}")
        
        # Parse JSON response
        response_text = result["message"].strip()
        
        # Clean JSON if needed
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json\n", "").replace("\n```", "")
        
        try:
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {response_text}")
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")
        
        # Create JDAnalysisResult
        analysis_result = JDAnalysisResult(
            job_id=job_id,
            required_skills=analysis_data.get("required_skills", []),
            nice_to_have_skills=analysis_data.get("nice_to_have_skills", []),
            experience_level=analysis_data.get("experience_level", "mid"),
            experience_years=analysis_data.get("experience_years"),
            soft_skills=analysis_data.get("soft_skills", []),
            industry_context=analysis_data.get("industry_context", ""),
            urgency_level=analysis_data.get("urgency_level", "medium"),
            cultural_requirements=analysis_data.get("cultural_requirements", []),
            education_requirements=analysis_data.get("education_requirements", []),
            certification_requirements=analysis_data.get("certification_requirements", []),
            work_arrangement=analysis_data.get("work_arrangement", "hybrid"),
            confidence=analysis_data.get("confidence", 0.8),
            raw_analysis=response_text
        )
        
        self.logger.info(f"JD analysis completed for job {job_id} with confidence {analysis_result.confidence}")
        
        return analysis_result
    
    async def analyze_job_by_id(self, job_id: str) -> JDAnalysisResult:
        """
        Convenience method to analyze job by ID
        
        Args:
            job_id: Job ID to analyze
            
        Returns:
            JDAnalysisResult
        """
        # Import here to avoid circular imports
        from function_tools import ai_tools
        
        # Get job data
        job_result = await ai_tools.get_job_data(job_id)
        if not job_result.get("success", False):
            raise Exception(f"Failed to get job data: {job_result.get('error', 'Unknown error')}")
        
        job_data = job_result["job_info"]
        
        # Analyze
        return await self._analyze_job_description(job_data)
