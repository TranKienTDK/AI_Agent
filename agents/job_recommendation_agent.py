"""
Job Recommendation Agent - Specialized agent for recommending jobs to candidates
Reuses existing agent architecture for CV-to-Jobs matching
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, date

from .base_agent import BaseAgent
from .agent_models import AgentMessage, MessageType, CVAnalysisResult, JDAnalysisResult
from .matching_agent import MatchingAgent

logger = logging.getLogger(__name__)

class JobRecommendationResult:
    """Result from Job Recommendation matching"""
    def __init__(self, cv_id: str, job_id: str, user_id: str, overall_score: float,
                 matched_skills: List[str], missing_skills: List[str], 
                 recommendation_reason: str):
        self.cv_id = cv_id
        self.job_id = job_id
        self.user_id = user_id
        self.overall_score = overall_score
        self.matched_skills = matched_skills
        self.missing_skills = missing_skills
        self.recommendation_reason = recommendation_reason
        self.created_date = date.today()

class JobRecommendationAgent(BaseAgent):
    """
    Specialized AI Agent for recommending jobs to candidates
    
    Responsibilities:
    - Analyze CV against multiple job opportunities
    - Rank jobs by compatibility score
    - Generate recommendation explanations
    - Prepare data for RecommendJob entity
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(agent_id="job_recommendation_agent", model_name=model_name)
        self.matching_agent = MatchingAgent()
        
    def get_system_prompt(self) -> str:
        return """
You are a specialized Career AI Agent expert in recommending job opportunities to candidates. Your role is to analyze a candidate's profile against multiple job opportunities and provide personalized job recommendations.

Your recommendation capabilities include:

1. **CV-Job Compatibility Analysis**:
   - Match candidate skills with job requirements
   - Assess experience level alignment
   - Evaluate career progression opportunities
   - Consider cultural and work style fit

2. **Job Ranking & Prioritization**:
   - Rank jobs by overall compatibility score
   - Consider candidate's career goals and aspirations
   - Balance current fit vs growth opportunities
   - Assess market competitiveness and opportunities

3. **Personalized Recommendations**:
   - Generate compelling recommendation reasons
   - Highlight skill matches and growth opportunities
   - Identify skill gaps and development areas
   - Provide actionable career advice

4. **Strategic Career Guidance**:
   - Consider career trajectory alignment
   - Evaluate learning and development opportunities
   - Assess company culture and work environment fit
   - Balance immediate needs vs long-term goals

**Scoring Framework** (0-100):
- 90-100: Perfect match, immediate application recommended
- 80-89: Excellent match, high priority opportunity
- 70-79: Good match, strong consideration
- 60-69: Moderate match, potential with skill development
- 50-59: Weak match, consider for future growth
- Below 50: Poor match, not recommended

**Recommendation Guidelines**:
- Focus on candidate's career growth and satisfaction
- Provide specific, actionable insights
- Balance optimism with realistic assessment
- Consider both technical and cultural fit
- Highlight unique opportunities and advantages

Always be encouraging while providing honest, evidence-based recommendations.
"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process job recommendation request
        
        Args:
            message: AgentMessage containing CV and job list data
            
        Returns:
            AgentMessage with job recommendation results
        """
        try:
            cv_analysis_data = message.data.get("cv_analysis")
            job_analyses_data = message.data.get("job_analyses", [])
            user_id = message.data.get("user_id")
            
            if not cv_analysis_data or not job_analyses_data:
                raise ValueError("Missing CV analysis or job analyses data")
            
            # Convert to structured objects
            cv_analysis = CVAnalysisResult(**cv_analysis_data)
            job_analyses = [JDAnalysisResult(**job_data) for job_data in job_analyses_data]
            
            self.logger.info(f"Processing job recommendations for CV {cv_analysis.cv_id} against {len(job_analyses)} jobs")
            
            # Generate recommendations for each job
            recommendations = await self._generate_job_recommendations(cv_analysis, job_analyses, user_id)
            
            # Sort by score (highest first)
            recommendations.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Create response
            return self._create_response_message(
                message_type=MessageType.ANALYSIS_RESULT,
                data={
                    "job_recommendations": [self._recommendation_to_dict(rec) for rec in recommendations],
                    "total_jobs_analyzed": len(job_analyses),
                    "recommended_jobs": len([r for r in recommendations if r.overall_score >= 50])
                },
                confidence=0.9,
                metadata={
                    "cv_id": cv_analysis.cv_id,
                    "user_id": user_id,
                    "processing_type": "job_recommendations"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing job recommendations: {str(e)}")
            return self._create_error_message(str(e), message)

    async def _generate_job_recommendations(self, 
                                          cv_analysis: CVAnalysisResult,
                                          job_analyses: List[JDAnalysisResult],
                                          user_id: str) -> List[JobRecommendationResult]:
        """
        Generate job recommendations for a candidate
        
        Args:
            cv_analysis: Analyzed CV data
            job_analyses: List of analyzed job data
            user_id: User ID for the candidate
            
        Returns:
            List of JobRecommendationResult objects
        """
        recommendations = []
        
        # Process jobs in batches for efficiency
        batch_size = 5
        for i in range(0, len(job_analyses), batch_size):
            batch = job_analyses[i:i + batch_size]
            batch_recommendations = await self._process_job_batch(cv_analysis, batch, user_id)
            recommendations.extend(batch_recommendations)
            
            # Small delay between batches
            if i + batch_size < len(job_analyses):
                await asyncio.sleep(0.5)
        
        return recommendations

    async def _process_job_batch(self,
                               cv_analysis: CVAnalysisResult,
                               job_analyses: List[JDAnalysisResult],
                               user_id: str) -> List[JobRecommendationResult]:
        """Process a batch of jobs for recommendation"""
        tasks = [
            self._analyze_job_match(cv_analysis, job_analysis, user_id)
            for job_analysis in job_analyses
        ]
        return await asyncio.gather(*tasks)

    async def _analyze_job_match(self,
                               cv_analysis: CVAnalysisResult,
                               job_analysis: JDAnalysisResult,
                               user_id: str) -> JobRecommendationResult:
        """
        Analyze match between CV and single job
        
        Args:
            cv_analysis: Analyzed CV data
            job_analysis: Analyzed job data
            user_id: User ID
            
        Returns:
            JobRecommendationResult
        """        # Create comprehensive analysis prompt
        analysis_prompt = f"""
Analyze this job opportunity for the candidate and provide detailed recommendation in JSON format.

**IMPORTANT: Use Semantic Skill Matching**
When matching skills, use semantic understanding and technical relationships, not just literal string matching:

**Common Skill Relationships:**
- ReactJS/React → JavaScript, JSX, Component-based development
- TypeScript → JavaScript, Static typing
- Angular → JavaScript, TypeScript, Component-based development  
- Vue.js → JavaScript, Component-based development
- Spring Framework → Java, Backend development, MVC
- Spring Boot → Java, Spring Framework, Microservices
- Node.js → JavaScript, Backend development
- Express.js → Node.js, JavaScript, Backend development
- Django → Python, Web framework, MVC
- Flask → Python, Web framework
- .NET/C# → Object-oriented programming, Microsoft ecosystem
- Laravel → PHP, Web framework, MVC
- React Native → React, JavaScript, Mobile development
- Flutter → Dart, Mobile development
- Kubernetes → Docker, Container orchestration, DevOps
- Docker → Containerization, DevOps
- AWS/Azure/GCP → Cloud computing, Infrastructure
- PostgreSQL/MySQL → SQL, Database management
- MongoDB → NoSQL, Database management
- Git → Version control
- Jenkins → CI/CD, DevOps
- Linux → System administration, Command line

**Candidate Profile:**
- Skills: {cv_analysis.skills}
- Skill Levels: {cv_analysis.skill_levels}
- Experience Years: {cv_analysis.experience_years}
- Career Level: {cv_analysis.career_level}
- Strengths: {cv_analysis.strengths}
- Career Progression: {cv_analysis.career_progression}
- Growth Potential: {cv_analysis.growth_potential}
- Work Preference: {cv_analysis.work_preference}

**Job Opportunity:**
- Required Skills: {job_analysis.required_skills}
- Nice-to-Have Skills: {job_analysis.nice_to_have_skills}
- Experience Level: {job_analysis.experience_level}
- Experience Years: {job_analysis.experience_years}
- Soft Skills: {job_analysis.soft_skills}
- Industry Context: {job_analysis.industry_context}
- Work Arrangement: {job_analysis.work_arrangement}
- Cultural Requirements: {job_analysis.cultural_requirements}

**Required JSON Output Format:**
{{
    "overall_score": 0-100,
    "matched_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill3", "skill4", ...],
    "semantically_matched_skills": ["skill_implied_by_candidate_skills", ...],
    "skill_match_percentage": 0-100,
    "experience_alignment": "excellent|good|moderate|poor",
    "career_growth_potential": "high|medium|low",
    "cultural_fit_assessment": "excellent|good|moderate|poor",
    "recommendation_reason": "Detailed explanation why this job is/isn't recommended",
    "key_highlights": ["highlight1", "highlight2", ...],
    "development_opportunities": ["opportunity1", "opportunity2", ...],
    "potential_challenges": ["challenge1", "challenge2", ...],
    "confidence": 0.0-1.0
}}

**Skill Matching Instructions:**
1. **Direct Matches**: Include skills that exactly match between candidate and job requirements
2. **Semantic Matches**: Include skills that are semantically related (e.g., if candidate has "React" and job needs "JavaScript", count JavaScript as matched)
3. **Implied Skills**: Add to "semantically_matched_skills" any job requirements that are covered by candidate's related skills
4. **Missing Skills**: Only include truly missing skills that cannot be inferred from candidate's existing skillset
5. **Skill Match Percentage**: Calculate based on both direct and semantic matches

**Analysis Guidelines:**
1. Focus on candidate's career growth and satisfaction
2. Provide specific evidence for recommendations considering semantic skill relationships
3. Consider both immediate fit and future potential
4. Be encouraging while realistic about challenges
5. Highlight unique opportunities this role offers
6. When explaining matches, mention semantic relationships (e.g., "Your React experience demonstrates JavaScript proficiency")

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
            self.logger.error(f"Failed to parse JSON response for job {job_analysis.job_id}: {response_text}")
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")
          # Validate and ensure skills are lists
        matched_skills = analysis_data.get("matched_skills", [])
        if isinstance(matched_skills, str):
            matched_skills = [skill.strip() for skill in matched_skills.split(",") if skill.strip()]
        elif not isinstance(matched_skills, list):
            matched_skills = []
            
        missing_skills = analysis_data.get("missing_skills", [])
        if isinstance(missing_skills, str):
            missing_skills = [skill.strip() for skill in missing_skills.split(",") if skill.strip()]
        elif not isinstance(missing_skills, list):
            missing_skills = []
            
        # Handle semantic matches - combine with direct matches for overall matched_skills
        semantically_matched_skills = analysis_data.get("semantically_matched_skills", [])
        if isinstance(semantically_matched_skills, str):
            semantically_matched_skills = [skill.strip() for skill in semantically_matched_skills.split(",") if skill.strip()]
        elif not isinstance(semantically_matched_skills, list):
            semantically_matched_skills = []
        
        # Combine direct and semantic matches (remove duplicates)
        all_matched_skills = list(set(matched_skills + semantically_matched_skills))          # Create JobRecommendationResult
        recommendation = JobRecommendationResult(
            cv_id=cv_analysis.cv_id,
            job_id=job_analysis.job_id,
            user_id=user_id,
            overall_score=analysis_data.get("overall_score", 0),
            matched_skills=all_matched_skills,  # Use combined matched skills (direct + semantic)
            missing_skills=missing_skills,
            recommendation_reason=analysis_data.get("recommendation_reason", "")
        )        
        self.logger.info(f"Job recommendation completed for job {job_analysis.job_id} with score {recommendation.overall_score}")
        self.logger.debug(f"Direct matches: {matched_skills}")
        self.logger.debug(f"Semantic matches: {semantically_matched_skills}")
        self.logger.debug(f"Combined matches: {all_matched_skills}")
        
        return recommendation

    def _recommendation_to_dict(self, recommendation: JobRecommendationResult) -> Dict[str, Any]:
        """Convert JobRecommendationResult to dictionary"""
        # Ensure skills are properly formatted as lists
        matched_skills = recommendation.matched_skills
        if isinstance(matched_skills, str):
            matched_skills = [skill.strip() for skill in matched_skills.split(",") if skill.strip()]
        elif not isinstance(matched_skills, list):
            matched_skills = []
            
        missing_skills = recommendation.missing_skills
        if isinstance(missing_skills, str):
            missing_skills = [skill.strip() for skill in missing_skills.split(",") if skill.strip()]
        elif not isinstance(missing_skills, list):
            missing_skills = []
            
        return {
            "cv_id": recommendation.cv_id,
            "job_id": recommendation.job_id,
            "user_id": recommendation.user_id,
            "overall_score": recommendation.overall_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "recommendation_reason": recommendation.recommendation_reason,
            "created_date": recommendation.created_date.isoformat()
        }
