"""
CV Analyzer Agent - Specialized agent for analyzing CVs
"""
import json
import logging
import asyncio
from typing import Dict, Any, List

from .base_agent import BaseAgent
from .agent_models import AgentMessage, MessageType, CVAnalysisResult

logger = logging.getLogger(__name__)

class CVAnalyzerAgent(BaseAgent):
    """
    Specialized AI Agent for analyzing CVs
    
    Responsibilities:
    - Extract and categorize skills with proficiency levels
    - Analyze career progression and stability
    - Identify strengths, weaknesses, and potential
    - Detect red flags and unique selling points
    - Assess growth potential and suitability
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(agent_id="cv_analyzer", model_name=model_name)
        
    def get_system_prompt(self) -> str:
        return """
You are a specialized HR AI Agent expert in analyzing CVs and candidate profiles. Your role is to extract comprehensive, structured insights about candidates to enable precise job matching.

Your analysis capabilities include:

1. **Technical Skills Assessment**:
   - Extract specific technical skills with accurate names
   - Infer skill proficiency levels from context (Beginner/Intermediate/Advanced)
   - Identify skill categories (Frontend, Backend, DevOps, etc.)
   - Recognize technology stacks and complementary skills

2. **Experience Analysis**:
   - Calculate total years of experience accurately
   - Determine career level: junior (0-2 years), mid (2-5 years), senior (5+ years)
   - Analyze career progression patterns (promotions, role evolution)
   - Assess industry diversity and domain expertise

3. **Strengths & Weaknesses Identification**:
   - Technical strengths based on depth of experience
   - Leadership and management capabilities
   - Problem-solving and innovation indicators
   - Communication and collaboration evidence
   - Areas needing development or skill gaps

4. **Red Flags Detection**:
   - Frequent job changes without clear progression
   - Extended unemployment gaps without explanation
   - Inconsistent information or timeline conflicts
   - Overqualification or underqualification signals
   - Missing expected skills for stated roles

5. **Unique Selling Points**:
   - Rare skill combinations
   - Impressive achievements or recognition
   - Unique background or perspective
   - Notable projects or contributions
   - Cross-functional or interdisciplinary experience

6. **Growth Potential Assessment**:
   - Learning agility indicators
   - Adaptability to new technologies
   - Career trajectory analysis
   - Skill development patterns
   - Potential for advancement

7. **Cultural & Soft Skills**:
   - Team collaboration indicators
   - Leadership style and experience
   - Communication abilities
   - Work style preferences
   - Cultural adaptability

**Analysis Framework**:
- Quantify experience by aggregating role durations
- Infer skills from project descriptions and technologies used
- Assess progression through role titles and responsibilities
- Identify patterns in career choices and development
- Balance positive indicators with potential concerns

**Output Requirements**:
- Return valid JSON with comprehensive analysis
- Include confidence scores for assessments
- Provide specific evidence for claims
- Flag uncertain or incomplete information
- Focus on actionable insights for recruitment decisions

Always be objective, thorough, and evidence-based in your analysis.
"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process CV analysis request (can handle single CV or batch)
        
        Args:
            message: AgentMessage containing CV data or CV IDs
            
        Returns:
            AgentMessage with CV analysis results
        """
        try:
            cv_data_list = message.data.get("cv_data_list", [])
            single_cv_data = message.data.get("cv_data")
            cv_ids = message.data.get("cv_ids", [])
              # Handle cv_ids by fetching CV data
            if cv_ids and not cv_data_list and not single_cv_data:
                from function_tools import ai_tools
                cv_data_list = []
                
                for cv_id in cv_ids:
                    cv_result = await ai_tools.get_raw_cv_data(cv_id)
                    if cv_result.get("success", False):
                        cv_data_list.append(cv_result["data"])
                    else:
                        self.logger.warning(f"Failed to get CV data for {cv_id}: {cv_result.get('error', 'Unknown error')}")
            
            if single_cv_data:
                # Single CV analysis
                cv_data_list = [single_cv_data]
            
            if not cv_data_list:
                raise ValueError("No CV data provided in message")
            
            self.logger.info(f"Processing CV analysis for {len(cv_data_list)} CVs")
            
            # Process CVs in batch
            analysis_results = await self._analyze_cvs_batch(cv_data_list)
            
            # Create response
            return self._create_response_message(
                message_type=MessageType.ANALYSIS_RESULT,
                data={"cv_analyses": [result.dict() for result in analysis_results]},
                confidence=sum(r.confidence for r in analysis_results) / len(analysis_results),
                metadata={
                    "total_cvs": len(cv_data_list),
                    "analysis_type": "cv_analysis",
                    "batch_processing": len(cv_data_list) > 1
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing CV analysis: {str(e)}")
            return self._create_error_message(str(e), message)
    
    async def _analyze_cvs_batch(self, cv_data_list: List[Dict[str, Any]]) -> List[CVAnalysisResult]:
        """
        Analyze multiple CVs in batch for efficiency
        
        Args:
            cv_data_list: List of CV data dictionaries
            
        Returns:
            List of CVAnalysisResult objects
        """
        # For small batches, process in parallel
        if len(cv_data_list) <= 5:
            tasks = [self._analyze_single_cv(cv_data) for cv_data in cv_data_list]
            return await asyncio.gather(*tasks)
        
        # For larger batches, process in chunks to avoid overwhelming LLM
        results = []
        chunk_size = 5
        
        for i in range(0, len(cv_data_list), chunk_size):
            chunk = cv_data_list[i:i + chunk_size]
            tasks = [self._analyze_single_cv(cv_data) for cv_data in chunk]
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
            
            # Small delay between chunks to be respectful to API
            if i + chunk_size < len(cv_data_list):
                await asyncio.sleep(0.5)
        
        return results
    
    async def _analyze_single_cv(self, cv_data: Dict[str, Any]) -> CVAnalysisResult:
        """
        Analyze a single CV using LLM
        
        Args:
            cv_data: Raw CV data from API
            
        Returns:
            CVAnalysisResult with structured analysis
        """
        # Extract CV information
        cv_id = cv_data.get("id", "")
        profile = cv_data.get("profile", "")
        skills = cv_data.get("skills", [])
        experiences = cv_data.get("experiences", [])
        projects = cv_data.get("projects", [])
        educations = cv_data.get("educations", [])
        certifications = cv_data.get("certifications", [])
        languages = cv_data.get("languages", [])
        
        # Format experience and project information
        experience_text = self._format_experiences(experiences)
        project_text = self._format_projects(projects)
        education_text = self._format_educations(educations)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
Analyze this candidate's CV and return a comprehensive structured analysis in JSON format.

**Candidate Information:**
- Profile Summary: {profile}
- Listed Skills: {skills}
- Work Experience: {experience_text}
- Projects: {project_text}
- Education: {education_text}
- Certifications: {certifications}
- Languages: {languages}

**Required JSON Output Format:**
{{
    "skills": ["skill1", "skill2", ...],
    "skill_levels": {{"skill1": "Beginner|Intermediate|Advanced", ...}},
    "experience_years": number or null,
    "career_level": "junior|mid|senior",
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "red_flags": ["flag1", "flag2", ...],
    "unique_selling_points": ["point1", "point2", ...],
    "career_progression": "description of career trajectory",
    "stability_score": 0.0-1.0,
    "growth_potential": "low|medium|high",
    "education_background": ["degree1", "degree2", ...],
    "certifications": ["cert1", "cert2", ...],
    "work_preference": "remote|hybrid|onsite",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of key assessments"
}}

**Analysis Guidelines:**
1. Extract all technical skills mentioned or implied from experience/projects
2. Calculate total years of professional experience
3. Assess career level based on roles, responsibilities, and timeline
4. Identify clear strengths backed by evidence
5. Note any concerning patterns or gaps
6. Highlight unique aspects that set candidate apart
7. Evaluate career progression and stability patterns
8. Assess potential for growth based on learning patterns

**Skill Level Assessment:**
- Beginner: Mentioned but limited evidence of depth
- Intermediate: Used in projects/work with some complexity
- Advanced: Deep experience, leadership, or specialized use

**Red Flag Examples:**
- Job hopping without clear progression
- Unexplained employment gaps
- Mismatched skills vs experience
- Inconsistent timeline or information

Return ONLY the JSON object, no additional text.
"""

        # Call LLM for analysis
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": analysis_prompt}
        ]
        
        result = await self._call_llm(messages, max_tokens=2000)
        
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
            self.logger.error(f"Failed to parse JSON response for CV {cv_id}: {response_text}")
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")
        
        # Create CVAnalysisResult
        analysis_result = CVAnalysisResult(
            cv_id=cv_id,
            skills=analysis_data.get("skills", []),
            skill_levels=analysis_data.get("skill_levels", {}),
            experience_years=analysis_data.get("experience_years"),
            career_level=analysis_data.get("career_level", "mid"),
            strengths=analysis_data.get("strengths", []),
            weaknesses=analysis_data.get("weaknesses", []),
            red_flags=analysis_data.get("red_flags", []),
            unique_selling_points=analysis_data.get("unique_selling_points", []),
            career_progression=analysis_data.get("career_progression", ""),
            stability_score=analysis_data.get("stability_score", 0.5),
            growth_potential=analysis_data.get("growth_potential", "medium"),
            education_background=analysis_data.get("education_background", []),
            certifications=analysis_data.get("certifications", []),
            work_preference=analysis_data.get("work_preference", "hybrid"),
            confidence=analysis_data.get("confidence", 0.8),
            raw_analysis=response_text
        )
        
        self.logger.info(f"CV analysis completed for {cv_id} with confidence {analysis_result.confidence}")
        
        return analysis_result
    
    def _format_experiences(self, experiences: List[Dict[str, Any]]) -> str:
        """Format experience data for analysis"""
        if not experiences:
            return "No work experience provided"
        
        formatted = []
        for exp in experiences:
            title = exp.get("title", "Unknown")
            company = exp.get("company", "Unknown")
            duration = f"{exp.get('startDate', 'Unknown')} - {exp.get('endDate', 'Present')}"
            description = exp.get("description", "No description")
            
            formatted.append(f"• {title} at {company} ({duration}): {description}")
        
        return "\n".join(formatted)
    
    def _format_projects(self, projects: List[Dict[str, Any]]) -> str:
        """Format project data for analysis"""
        if not projects:
            return "No projects provided"
        
        formatted = []
        for proj in projects:
            if isinstance(proj, dict):
                name = proj.get("project", "Unknown Project")
                description = proj.get("description", "No description")
                duration = f"{proj.get('startDate', '')} - {proj.get('endDate', '')}"
                
                formatted.append(f"• {name} ({duration}): {description}")
        
        return "\n".join(formatted) if formatted else "No valid projects found"
    
    def _format_educations(self, educations: List[Dict[str, Any]]) -> str:
        """Format education data for analysis"""
        if not educations:
            return "No education provided"
        
        formatted = []
        for edu in educations:
            degree = edu.get("degree", "Unknown")
            field = edu.get("field", "Unknown")
            school = edu.get("school", "Unknown")
            year = edu.get("graduationYear", "Unknown")
            
            formatted.append(f"• {degree} in {field} from {school} ({year})")
        
        return "\n".join(formatted)
    
    async def analyze_cv_by_id(self, cv_id: str) -> CVAnalysisResult:
        """
        Convenience method to analyze CV by ID
        
        Args:
            cv_id: CV ID to analyze
            
        Returns:
            CVAnalysisResult
        """
        # Import here to avoid circular imports
        from function_tools import ai_tools
        
        # Get CV data
        cv_result = await ai_tools.get_cv_data(cv_id)
        if not cv_result.get("success", False):
            raise Exception(f"Failed to get CV data: {cv_result.get('error', 'Unknown error')}")
        
        cv_data = cv_result["data"]
        
        # Analyze
        return await self._analyze_single_cv(cv_data)
