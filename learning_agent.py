import asyncio
import json
import logging
import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
from gemini_service import normalize_data, calculate_relevance
from models import CvInput, JdInput, CvMatchResult
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI app
app = FastAPI()

# Spring Boot API base URL
SPRING_BOOT_API = "http://localhost:8080"

# Global skill weights
skill_weights: Dict[str, float] = {}

async def train_skill_weights():
    global skill_weights
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SPRING_BOOT_API}/evaluations")
            response.raise_for_status()
            evaluations = response.json()

        if len(evaluations) < 10:
            logger.info("Not enough data to train skill weights")
            return

        cv_data = []
        for eval in evaluations:
            feedback = eval.get('feedback')
            if feedback not in ['approved', 'rejected']:
                continue
            skills = eval.get('skills', [])
            cv_data.append((skills, 1 if feedback == 'approved' else 0))

        if not cv_data:
            logger.info("No valid feedback data for training")
            return

        all_skills = set()
        for skills, _ in cv_data:
            all_skills.update(skills)
        all_skills = sorted(list(all_skills))

        X = np.zeros((len(cv_data), len(all_skills)))
        y = []
        for i, (skills, label) in enumerate(cv_data):
            for skill in skills:
                if skill in all_skills:
                    X[i, all_skills.index(skill)] = 1
            y.append(label)

        model = LogisticRegression()
        model.fit(X, y)

        skill_weights.clear()
        for skill, coef in zip(all_skills, model.coef_[0]):
            skill_weights[skill] = max(0.1, coef)
        logger.info(f"Trained skill weights: {skill_weights}")

    except Exception as e:
        logger.error(f"Error training skill weights: {str(e)}")

async def save_evaluation(cv_id: str, job_id: str, apply_id: Optional[str], score: float, skills: List[str], explanation: str, feedback: str = None):
    try:
        async with httpx.AsyncClient() as client:
            evaluation = {
                "cvId": cv_id,
                "jobId": job_id,
                "score": score,
                "explanation": explanation,
                "skills": skills,
                "feedback": feedback
            }
            response = await client.post(f"{SPRING_BOOT_API}/evaluations", json=evaluation)
            response.raise_for_status()
            evaluation_id = response.json()['id']
            logger.info(f"Saved evaluation for CV {cv_id}, Job {job_id}: score={score}, skills={skills}")

            if apply_id:
                response = await client.patch(f"{SPRING_BOOT_API}/apply/{apply_id}/evaluation", json=evaluation_id)
                response.raise_for_status()
                logger.info(f"Updated Apply {apply_id} with evaluationId {evaluation_id}")

    except Exception as e:
        logger.error(f"Error saving evaluation: {str(e)}")

@app.post("/sync_feedback")
async def sync_feedback(evaluation_id: str = Query(...), feedback: str = Body(...)):
    try:
        logger.info(f"Received feedback for evaluation {evaluation_id}: {feedback}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8080/evaluations")
            response.raise_for_status()
            evaluations = response.json()["data"]

        X = []
        y = []
        for eval in evaluations:
            if eval["feedback"] not in ["approved", "rejected"]:
                continue
            cv_id = eval["cvId"]
            skills = eval["skills"]
            skill_embeddings = sbert_model.encode(skills, convert_to_tensor=True)
            X.append(skill_embeddings.mean(dim=0).numpy())
            y.append(1 if eval["feedback"] == "approved" else 0)

        if len(X) > 0:
            model = LogisticRegression()
            model.fit(X, y)
            joblib.dump(model, 'skill_weights.joblib')
            logger.info("Model retrained successfully")
        
        return {"message": "Feedback processed successfully"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def match_cvs_with_learning(cvs: List[CvInput], jd: JdInput, apply_ids: List[Optional[str]]) -> List[CvMatchResult]:
    """Match CVs against JD and return ranked results"""
    try:
        if not cvs:
            logger.error("Empty CV list provided")
            raise ValueError("Empty CV list provided")
        if len(cvs) != len(apply_ids):
            logger.error("Mismatch between CVs and apply_ids")
            raise ValueError("Number of CVs and apply_ids must match")

        # Train skill weights
        await train_skill_weights()

        # Normalize JD data
        logger.info("Normalizing JD data")
        jd_data = await normalize_data(jd.dict(), "JD")
        logger.info(f"JD data: {jd_data}")

        # Encode JD skills with SBERT
        jd_skill_embeddings = sbert_model.encode(jd_data.skills, convert_to_tensor=True)

        results = []
        for cv, apply_id in zip(cvs, apply_ids):
            try:
                logger.info(f"Processing CV: {cv.cv_id}")
                cv_data = await normalize_data(cv.dict(), "CV")
                logger.info(f"CV {cv.cv_id} data: {cv_data}")

                # Apply business rules with SBERT
                mandatory_skills = set(jd_data.skills)
                cv_skills = set(cv_data.skills)
                matched_skills = set()

                # Semantic skill matching with SBERT
                cv_skill_embeddings = sbert_model.encode(cv_data.skills, convert_to_tensor=True)
                similarity_matrix = util.cos_sim(cv_skill_embeddings, jd_skill_embeddings)
                for i, cv_skill in enumerate(cv_data.skills):
                    for j, jd_skill in enumerate(jd_data.skills):
                        if similarity_matrix[i][j] > 0.8:
                            matched_skills.add(jd_skill)

                missing_skills = mandatory_skills - matched_skills
                skill_match_ratio = len(matched_skills) / len(mandatory_skills) if mandatory_skills else 1.0
                if skill_match_ratio < 0.6:
                    logger.info(f"CV {cv.cv_id} rejected: Only {skill_match_ratio:.2%} mandatory skills matched")
                    explanation = f"Rejected: Only {len(matched_skills)}/{len(mandatory_skills)} mandatory skills matched ({skill_match_ratio:.2%}). Missing skills: {', '.join(missing_skills)}"
                    results.append(CvMatchResult(cv_id=cv.cv_id, score=0.0, explanation=explanation))
                    await save_evaluation(cv.cv_id, jd.job_id, apply_id, 0.0, cv_data.skills, explanation)
                    continue

                jd_years = jd_data.years_experience or 0
                cv_years = cv_data.years_experience or 0
                if jd_years > 0 and cv_years < jd_years * 0.8:
                    logger.info(f"CV {cv.cv_id} rejected: Insufficient experience ({cv_years} vs {jd_years} years)")
                    explanation = f"Rejected: Insufficient experience ({cv_years} vs {jd_years} years)"
                    results.append(CvMatchResult(cv_id=cv.cv_id, score=0.0, explanation=explanation))
                    await save_evaluation(cv.cv_id, jd.job_id, apply_id, 0.0, cv_data.skills, explanation)
                    continue

                # Calculate relevance with learned skill weights
                gemini_score, gemini_explanation = await calculate_relevance(cv_data, jd_data)
                skill_score = sum(skill_weights.get(skill, 0.5) for skill in matched_skills) / sum(skill_weights.get(skill, 0.5) for skill in mandatory_skills) * 40
                experience_score = min(cv_data.years_experience / jd_data.years_experience, 1.0) * 30 if jd_data.years_experience else 30
                education_score = 20 if cv_data.education and jd_data.education.lower() in cv_data.education.lower() else 0
                certification_score = len(set(cv_data.certifications).intersection(jd_data.certifications)) / len(jd_data.certifications) * 10 if jd_data.certifications else 10
                final_score = (gemini_score * 0.5) + (skill_score + experience_score + education_score + certification_score) * 0.5

                explanation = (
                    f"Skills: Matched {len(matched_skills)}/{len(mandatory_skills)} mandatory skills ({skill_match_ratio:.2%}); "
                    f"Experience: CV {cv_years:.1f} years vs JD {jd_years:.1f} years ({experience_score/30*100:.2f}% match); "
                    f"Education: {'Matched' if education_score > 0 else 'Not matched'}; "
                    f"Certifications: Matched {len(set(cv_data.certifications).intersection(jd_data.certifications))}/{len(jd_data.certifications)} certifications ({certification_score/10*100:.2f}%)"
                )                logger.info(f"CV {cv.cv_id} final score: {final_score}")
                results.append(CvMatchResult(cv_id=cv.cv_id, score=final_score, explanation=explanation))
                await save_evaluation(cv.cv_id, jd.job_id, apply_id, final_score, cv_data.skills, explanation)

            except Exception as e:
                logger.error(f"Error processing CV {cv.cv_id}: {str(e)}")
                explanation = f"Error: {str(e)}"
                results.append(CvMatchResult(cv_id=cv.cv_id, score=0.0, explanation=explanation))
                await save_evaluation(cv.cv_id, jd.job_id, apply_id, 0.0, cv_data.skills, explanation)

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Returning {len(results)} results")
        return results    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise

if __name__ == "__main__":
    pass