from fastapi import HTTPException
from typing import List, Dict, Any
import logging
from openai_service import ai_agent_chat
from pydantic import BaseModel
from models import ActionRecommendation, AIAgentRecommendation

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class AIAgentRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    use_functions: bool = True

class AIAgentResponse(BaseModel):
    success: bool
    message: str
    function_calls: List[Dict[str, Any]] = []
    has_function_calls: bool = False
    error: str = None

class AIAgentController:
    """
    Controller cho AI Agent với function calling capabilities
    """
    
    def __init__(self):        self.system_prompt = """
You are an intelligent HR AI Agent specialized in CV-JD matching and recruitment analysis. 

Your capabilities include:
1. **CV Management**: Retrieve and analyze CV data from the system
2. **Job Analysis**: Access job descriptions and requirements
3. **Matching & Evaluation**: Perform comprehensive CV-JD matching with detailed scoring
4. **Action Recommendations**: Suggest appropriate HR actions based on analysis

**Available Functions:**
- get_cv_data(cv_id): Get specific CV or all CVs
- get_job_data(job_id): Get job description details
- match_cv_with_job(cv_id, job_id): Perform CV-JD matching analysis
- recommend_actions(cv_id, job_id, score, explanation): Generate action recommendations
- save_evaluation(cv_id, job_id, score, explanation): Save matching results (only when explicitly requested)

**IMPORTANT GUIDELINES:**
- ALWAYS use recommend_actions() instead of automatically executing actions
- HR personnel must make the final decisions on all actions
- Only use save_evaluation() when HR explicitly requests to save results
- Provide clear reasoning for each recommendation
- Focus on advisory role, not execution

**Available Action Types (Only 2 types):**
1. "send_contact_email": For highly potential candidates (score ≥70%) who should be contacted immediately
2. "save_cv": For moderately potential candidates (score ≥50%) who should be saved for future consideration

**Scoring Thresholds:**
- Score ≥70%: Recommend "send_contact_email" 
- Score ≥50%: Recommend "save_cv"
- Score <50%: Recommend "no_action"

**Instructions:**
- Always provide detailed, professional analysis
- Use function calls when users request data retrieval or analysis
- Generate recommendations using recommend_actions() function
- Explain your reasoning and methodology clearly
- Provide actionable insights with clear next steps
- Let HR make the final decisions based on your suggestions

**Example Workflow:**
1. User: "Analyze CV 123 for job 456"
2. You: Call match_cv_with_job(123, 456) → get score & explanation
3. You: Call recommend_actions(123, 456, score, explanation) → generate recommendations
4. You: Present analysis and recommendations to HR for decision

Respond professionally and use function calls when appropriate to provide accurate, real-time data.
"""
    
    async def process_request(self, request: AIAgentRequest) -> AIAgentResponse:
        """
        Process AI Agent request with function calling
        
        Args:
            request: AIAgentRequest containing user message and context
            
        Returns:
            AIAgentResponse with AI response and function call results
        """
        try:
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            for msg in request.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Add current user message
            messages.append({"role": "user", "content": request.message})
            
            # Get AI response with function calling
            result = await ai_agent_chat(messages, use_functions=request.use_functions)
            
            if result["success"]:
                return AIAgentResponse(
                    success=True,
                    message=result["message"],
                    function_calls=result["function_calls"],
                    has_function_calls=result["has_function_calls"]
                )
            else:
                return AIAgentResponse(
                    success=False,
                    message="Sorry, I encountered an error processing your request.",
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Error processing AI agent request: {str(e)}")
            return AIAgentResponse(
                success=False,
                message="Sorry, I encountered an unexpected error.",
                error=str(e)
            )
    
    async def quick_match(self, cv_id: str, job_id: str, save_result: bool = True) -> AIAgentResponse:
        """
        Quick CV-JD matching shortcut
        
        Args:
            cv_id: ID of CV to match
            job_id: ID of job to match against
            save_result: Whether to save the evaluation result
            
        Returns:
            AIAgentResponse with matching results
        """
        try:
            message = f"Perform a detailed CV-JD matching analysis for CV {cv_id} and Job {job_id}."
            if save_result:
                message += " Save the evaluation results to the system."
                
            request = AIAgentRequest(
                message=message,
                use_functions=True
            )
            
            return await self.process_request(request)
            
        except Exception as e:
            logger.error(f"Error in quick match: {str(e)}")
            return AIAgentResponse(
                success=False,
                message="Failed to perform quick match.",
                error=str(e)
            )
    
    async def analyze_all_candidates(self, job_id: str, top_n: int = 5) -> AIAgentResponse:
        """
        Analyze all candidates for a specific job
        
        Args:
            job_id: ID of job to analyze candidates for
            top_n: Number of top candidates to highlight
            
        Returns:
            AIAgentResponse with analysis results
        """
        try:
            message = f"""
Analyze all available candidates for job {job_id}. Please:
1. Get all CV data from the system
2. Get the job description for job {job_id}
3. Perform matching analysis for each candidate
4. Rank candidates by match score
5. Provide detailed insights for the top {top_n} candidates
6. Save evaluation results for all candidates

Provide a comprehensive report with recommendations.
"""
            
            request = AIAgentRequest(
                message=message,
                use_functions=True
            )
            return await self.process_request(request)
            
        except Exception as e:
            logger.error(f"Error analyzing all candidates: {str(e)}")
            return AIAgentResponse(
                success=False,
                message="Failed to analyze candidates.",
                error=str(e)
            )
    
    def _generate_action_recommendations(self, score: float, explanation: str, cv_id: str, job_id: str) -> List[ActionRecommendation]:
        """
        Generate action recommendations based on matching score and analysis
        
        Args:
            score: Matching score (0-100)
            explanation: Match explanation
            cv_id: CV ID
            job_id: Job ID
            
        Returns:
            List of ActionRecommendation objects
        """        
        recommendations = []
        
        if score >= 70:
            # Ứng viên rất tiềm năng - gửi email liên hệ
            recommendations.append(ActionRecommendation(
                action_type="send_contact_email",
                priority="high",
                reason=f"Ứng viên rất tiềm năng với {score}% độ phù hợp. Nên liên hệ ngay để không bỏ lỡ nhân tài.",
                suggested_next_steps=[
                    "Gửi email liên hệ trong vòng 24 giờ",
                    "Sắp xếp cuộc gọi điện thoại sơ bộ",
                    "Chuẩn bị câu hỏi phỏng vấn"
                ]
            ))
        elif score >= 50:
            # Ứng viên khá tiềm năng - lưu CV
            recommendations.append(ActionRecommendation(
                action_type="save_cv",
                priority="medium",
                reason=f"Ứng viên khá tiềm năng với {score}% độ phù hợp. Nên lưu lại để xem xét cho các vị trí tương lai.",
                suggested_next_steps=[
                    "Lưu CV vào cơ sở dữ liệu nhân tài",
                    "Đánh dấu cho các vị trí tương tự trong tương lai",
                    "Theo dõi sự phát triển kỹ năng của ứng viên"
                ]
            ))
        # Note: No action recommended for scores < 50%, but still provide the evaluation data
            
        return recommendations
    
    async def get_recommendations_for_match(self, cv_id: str, job_id: str) -> AIAgentRecommendation:
        """
        Get AI recommendations for a specific CV-Job match
        
        Args:
            cv_id: CV ID
            job_id: Job ID
            
        Returns:
            AIAgentRecommendation with suggested actions
        """
        try:
            # Use existing match functionality
            message = f"Analyze CV {cv_id} for Job {job_id} and provide detailed matching results. Focus on providing clear analysis for HR decision making."
            
            request = AIAgentRequest(
                message=message,
                use_functions=True
            )
            
            result = await self.process_request(request)
            
            if result.success:
                # Extract score and explanation from function calls or AI response
                # This would need to be parsed from the AI response
                # For now, using placeholder values - in real implementation, 
                # you'd parse the actual results from the AI response
                
                score = 75.0  # This should be extracted from actual results
                explanation = result.message  # This should be the detailed explanation
                
                recommendations = self._generate_action_recommendations(score, explanation, cv_id, job_id)
                
                return AIAgentRecommendation(
                    cv_id=cv_id,
                    job_id=job_id,
                    match_score=score,
                    match_explanation=explanation,
                    recommended_actions=recommendations,
                    hr_notes="AI-generated recommendations. Please review and make final decision based on your professional judgment."
                )
            else:
                raise Exception(f"Failed to get match results: {result.error}")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

# Global controller instance
ai_agent_controller = AIAgentController()
