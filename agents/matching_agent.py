"""
Matching Agent - Specialized agent for CV-JD matching and evaluation
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Tuple

from .base_agent import BaseAgent
from .agent_models import (
    AgentMessage, MessageType, MatchingResult, 
    JDAnalysisResult, CVAnalysisResult
)

logger = logging.getLogger(__name__)

class MatchingAgent(BaseAgent):
    """
    Specialized AI Agent for matching CVs with Job Descriptions
    
    Responsibilities:
    - Perform multi-dimensional matching analysis
    - Calculate weighted scores across different criteria
    - Generate detailed explanations and insights
    - Provide actionable recommendations
    - Assess cultural fit and growth potential
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(agent_id="matching_agent", model_name=model_name)
        
        # Configurable scoring weights
        self.scoring_weights = {
            "skills": 0.4,
            "experience": 0.3,
            "education": 0.1,
            "cultural_fit": 0.1,
            "growth_potential": 0.1
        }
        
    def get_system_prompt(self) -> str:
        return """
You are a specialized HR AI Agent expert in matching candidates with job requirements. Your role is to perform comprehensive, multi-dimensional analysis to determine candidate-job fit and provide actionable insights.

Your matching capabilities include:

1. **Skills Matching Analysis**:
   - Match technical skills between CV and JD with precision
   - Assess skill proficiency levels vs requirements
   - Identify transferable and related skills
   - Calculate skill coverage and gap analysis
   - Weight must-have vs nice-to-have skills appropriately

2. **Experience Evaluation**:
   - Compare years of experience with requirements
   - Assess relevant domain/industry experience
   - Evaluate career progression alignment
   - Consider overqualification vs underqualification
   - Account for career transition potential

3. **Cultural & Soft Skills Fit**:
   - Match work style preferences (remote/hybrid/onsite)
   - Assess team collaboration indicators
   - Evaluate leadership alignment with role needs
   - Consider communication and interpersonal skills
   - Match company culture with candidate background

4. **Growth Potential Assessment**:
   - Evaluate learning agility and adaptability
   - Assess potential for skill development
   - Consider career trajectory alignment
   - Match growth mindset with role evolution needs
   - Balance current fit vs future potential

5. **Comprehensive Scoring Framework**:
   - Skills: 40% weight (technical competency)
   - Experience: 30% weight (proven capability)
   - Education: 10% weight (foundational knowledge)
   - Cultural Fit: 10% weight (team integration)
   - Growth Potential: 10% weight (future value)

6. **Risk & Opportunity Analysis**:
   - Identify potential red flags or concerns
   - Highlight unique strengths and differentiators
   - Assess retention likelihood
   - Evaluate onboarding complexity
   - Consider market competitiveness

**Matching Methodology**:
- Prioritize must-have requirements over nice-to-have
- Consider skill transferability and learning potential
- Balance immediate capability with growth trajectory
- Account for industry and domain expertise
- Weight recent experience more heavily
- Consider team dynamics and cultural alignment

**Scoring Guidelines**:
- 90-100: Exceptional match, immediate hire recommended
- 80-89: Strong match, high priority candidate
- 70-79: Good match, solid consideration
- 60-69: Moderate match, conditional consideration
- 50-59: Weak match, significant development needed
- Below 50: Poor match, not recommended

**Output Requirements**:
- Return valid JSON with detailed scoring breakdown
- Provide specific evidence for all assessments
- Include actionable recommendations
- Highlight both strengths and concerns
- Focus on practical hiring decisions

Always be objective, thorough, and evidence-based in your matching analysis.
"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process matching request (can handle single match or batch)
        
        Args:
            message: AgentMessage containing JD and CV analysis data
            
        Returns:
            AgentMessage with matching results
        """
        try:
            jd_analysis_data = message.data.get("jd_analysis")
            cv_analyses_data = message.data.get("cv_analyses", [])
            single_cv_analysis = message.data.get("cv_analysis")
            
            if single_cv_analysis:
                cv_analyses_data = [single_cv_analysis]
            
            if not jd_analysis_data or not cv_analyses_data:
                raise ValueError("Missing JD analysis or CV analyses data")
            
            # Convert to structured objects
            jd_analysis = JDAnalysisResult(**jd_analysis_data)
            cv_analyses = [CVAnalysisResult(**cv_data) for cv_data in cv_analyses_data]
            
            self.logger.info(f"Processing matching for job {jd_analysis.job_id} with {len(cv_analyses)} CVs")
            
            # Perform matching analysis
            matching_results = await self._match_candidates_batch(jd_analysis, cv_analyses)
            
            # Create response
            return self._create_response_message(
                message_type=MessageType.MATCHING_RESULT,
                data={"matching_results": [result.dict() for result in matching_results]},
                confidence=sum(r.confidence for r in matching_results) / len(matching_results),
                metadata={
                    "job_id": jd_analysis.job_id,
                    "total_matches": len(matching_results),
                    "matching_type": "cv_jd_matching"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing matching: {str(e)}")
            return self._create_error_message(str(e), message)
    
    async def _match_candidates_batch(self, 
                                    jd_analysis: JDAnalysisResult, 
                                    cv_analyses: List[CVAnalysisResult]) -> List[MatchingResult]:
        """
        Match multiple candidates with job requirements
        
        Args:
            jd_analysis: Analyzed job requirements
            cv_analyses: List of analyzed CV data
            
        Returns:
            List of MatchingResult objects
        """
        # For small batches, process in parallel
        if len(cv_analyses) <= 5:
            tasks = [self._match_single_candidate(jd_analysis, cv_analysis) 
                    for cv_analysis in cv_analyses]
            return await asyncio.gather(*tasks)
        
        # For larger batches, process in chunks
        results = []
        chunk_size = 5
        
        for i in range(0, len(cv_analyses), chunk_size):
            chunk = cv_analyses[i:i + chunk_size]
            tasks = [self._match_single_candidate(jd_analysis, cv_analysis) 
                    for cv_analysis in chunk]
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
            
            # Small delay between chunks
            if i + chunk_size < len(cv_analyses):
                await asyncio.sleep(0.5)
        
        return results
    
    async def _match_single_candidate(self, 
                                    jd_analysis: JDAnalysisResult, 
                                    cv_analysis: CVAnalysisResult) -> MatchingResult:
        """
        Match a single candidate with job requirements using LLM
        
        Args:
            jd_analysis: Analyzed job requirements
            cv_analysis: Analyzed CV data
            
        Returns:
            MatchingResult with detailed matching analysis
        """
        # Prepare matching prompt
        matching_prompt = f"""
Perform comprehensive matching analysis between this candidate and job requirements. Return detailed scoring and insights in JSON format.

**Job Requirements:**
- Required Skills: {jd_analysis.required_skills}
- Nice-to-Have Skills: {jd_analysis.nice_to_have_skills}
- Experience Level: {jd_analysis.experience_level}
- Experience Years: {jd_analysis.experience_years}
- Soft Skills: {jd_analysis.soft_skills}
- Industry Context: {jd_analysis.industry_context}
- Cultural Requirements: {jd_analysis.cultural_requirements}
- Education Requirements: {jd_analysis.education_requirements}
- Work Arrangement: {jd_analysis.work_arrangement}
- Urgency Level: {jd_analysis.urgency_level}

**Candidate Profile:**
- Skills: {cv_analysis.skills}
- Skill Levels: {cv_analysis.skill_levels}
- Experience Years: {cv_analysis.experience_years}
- Career Level: {cv_analysis.career_level}
- Strengths: {cv_analysis.strengths}
- Weaknesses: {cv_analysis.weaknesses}
- Red Flags: {cv_analysis.red_flags}
- Unique Selling Points: {cv_analysis.unique_selling_points}
- Career Progression: {cv_analysis.career_progression}
- Growth Potential: {cv_analysis.growth_potential}
- Education: {cv_analysis.education_background}
- Work Preference: {cv_analysis.work_preference}

**Required JSON Output Format:**
{{
    "overall_score": 0-100,
    "skill_match_score": 0-100,
    "experience_match_score": 0-100,
    "education_match_score": 0-100,
    "cultural_fit_score": 0-100,
    "growth_potential_score": 0-100,
    "matched_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill3", "skill4", ...],
    "transferable_skills": ["skill5", "skill6", ...],
    "skill_gaps": ["gap1", "gap2", ...],
    "strengths": ["strength1", "strength2", ...],
    "concerns": ["concern1", "concern2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "recommended_action": "send_contact_email|save_cv|no_action",
    "action_reason": "Detailed reason for recommendation",
    "priority_level": "low|medium|high",
    "confidence": 0.0-1.0,
    "explanation": "Comprehensive explanation of matching analysis",
    "detailed_breakdown": {{
        "skills_analysis": "Detailed skills comparison",
        "experience_analysis": "Experience fit assessment", 
        "cultural_fit_analysis": "Cultural alignment evaluation",
        "growth_assessment": "Growth potential evaluation",
        "risk_factors": "Potential concerns or risks",
        "unique_value": "What makes this candidate special"
    }}
}}

**Scoring Methodology:**
- Skills (40%): Match required skills, consider proficiency levels, account for transferable skills
- Experience (30%): Years alignment, domain relevance, career progression quality
- Education (10%): Degree requirements, field relevance, additional qualifications
- Cultural Fit (10%): Work style, team fit, communication alignment
- Growth Potential (10%): Learning ability, adaptability, career trajectory

**Action Recommendations:**
- send_contact_email: Score ≥70%, strong immediate fit
- save_cv: Score 50-69%, potential for future or with development
- no_action: Score <50%, poor fit

**Analysis Guidelines:**
1. Be strict with must-have requirements
2. Give credit for transferable and related skills
3. Consider overqualification vs underqualification carefully
4. Assess cultural and team fit realistically
5. Balance current capability with growth potential
6. Provide specific, actionable insights
7. Be honest about concerns while highlighting positives

Return ONLY the JSON object, no additional text.
"""

        # Call LLM for matching analysis
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": matching_prompt}
        ]
        
        result = await self._call_llm(messages, max_tokens=2500)
        
        if not result.get("success", False):
            raise Exception(f"LLM matching analysis failed: {result.get('error', 'Unknown error')}")
        
        # Parse JSON response
        response_text = result["message"].strip()
        
        # Clean JSON if needed
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json\n", "").replace("\n```", "")
        
        try:
            matching_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response for CV {cv_analysis.cv_id}: {response_text}")
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")
        
        # Create MatchingResult
        matching_result = MatchingResult(
            cv_id=cv_analysis.cv_id,
            job_id=jd_analysis.job_id,
            overall_score=matching_data.get("overall_score", 0),
            skill_match_score=matching_data.get("skill_match_score", 0),
            experience_match_score=matching_data.get("experience_match_score", 0),
            education_match_score=matching_data.get("education_match_score", 0),
            cultural_fit_score=matching_data.get("cultural_fit_score", 0),
            growth_potential_score=matching_data.get("growth_potential_score", 0),
            matched_skills=matching_data.get("matched_skills", []),
            missing_skills=matching_data.get("missing_skills", []),
            transferable_skills=matching_data.get("transferable_skills", []),
            skill_gaps=matching_data.get("skill_gaps", []),
            strengths=matching_data.get("strengths", []),
            concerns=matching_data.get("concerns", []),
            recommendations=matching_data.get("recommendations", []),
            recommended_action=matching_data.get("recommended_action"),
            action_reason=matching_data.get("action_reason", ""),
            priority_level=matching_data.get("priority_level", "medium"),
            confidence=matching_data.get("confidence", 0.8),
            explanation=matching_data.get("explanation", ""),
            detailed_breakdown=matching_data.get("detailed_breakdown", {})
        )
        
        self.logger.info(f"Matching completed for CV {cv_analysis.cv_id} with score {matching_result.overall_score}")
        
        return matching_result
    
    def update_scoring_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights for matching algorithm
        
        Args:
            new_weights: Dictionary of new weights
        """
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.scoring_weights.update(new_weights)
        self.logger.info(f"Updated scoring weights: {self.scoring_weights}")
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """Get current scoring weights"""
        return self.scoring_weights.copy()
