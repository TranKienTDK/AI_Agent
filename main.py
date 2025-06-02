import httpx
import websockets
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from models import CvInput, JdInput, CvMatchResult, ProjectInput, LanguageInput
from ai_agent import match_cvs_with_agent
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()
JWT_TOKEN = os.getenv("JWT_TOKEN")

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/match", response_model=List[CvMatchResult])
async def match_cvs(cvs: List[CvInput], jd: JdInput):
    try:
        results = await match_cvs_with_agent(cvs, jd)
        return results
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/match-all/{job_id}", response_model=List[CvMatchResult])
async def match_all_cvs(job_id: str, skill_filter: Optional[str] = None):
    try:
        headers = {"Authorization": f"Bearer {JWT_TOKEN}"}

        async with httpx.AsyncClient() as client:
            job_response = await client.get(f"http://localhost:8080/api/v1/job/{job_id}")
            job_response.raise_for_status()
            job_data = job_response.json()
            logger.info(f"JD response: {job_data}")
            description = job_data.get("data", {}).get("description", "") or "No job description provided"
            jd = JdInput(
                required_skills=job_data.get("data", {}).get("skillNames", []),
                required_experience=description,
                required_education="",
                required_certifications=[],
                text=description
            )

        async with httpx.AsyncClient() as client:
            cv_response = await client.get("http://localhost:8080/api/v1/cv/all", headers=headers)
            cv_response.raise_for_status()
            cv_data = cv_response.json()["data"]
            logger.info(f"CV data: {cv_data}")
            cvs = [
                CvInput(
                    cv_id=cv["id"],
                    skills=[skill["name"] for skill in cv.get("skills", [])],
                    experience="; ".join([exp["description"] for exp in cv.get("experiences", [])]),
                    education="; ".join([edu["field"] + ": " + edu.get("description", "") for edu in cv.get("educations", [])]),
                    certifications=[cert["certificate"] for cert in cv.get("certifications", [])],
                    projects=[
                        ProjectInput(
                            project=p["project"],
                            description=p.get("description", ""),
                            start_date=p.get("startDate", ""),
                            end_date=p.get("endDate", "")
                        ) for p in cv.get("projects", []) if p is not None and p.get("project") is not None and isinstance(p.get("project"), str)
                    ],
                    languages=[
                        LanguageInput(
                            language=l["language"],
                            level=l.get("level", "") if l.get("level") is not None else ""
                        ) for l in cv.get("languages", []) if l is not None and l.get("language") is not None and isinstance(l.get("language"), str)
                    ],
                    text=cv.get("profile", "") + " " + cv.get("additionalInfo", ""),
                    email=cv.get("info", {}).get("email", ""),
                    phone=cv.get("info", {}).get("phone", "")
                ) for cv in cv_data
            ]

        results = await match_cvs_with_agent(cvs, jd)

        async with httpx.AsyncClient() as client:
            for result in results:
                evaluation = {
                    "cvId": result.cv_id,
                    "jobId": job_id,
                    "score": result.score,
                    "explanation": result.explanation,
                    "skills": next(cv["skills"] for cv in cv_data if cv["id"] == result.cv_id),
                    "feedback": None
                }
                await client.post("http://localhost:8080/api/v1/evaluations", json=evaluation, headers=headers)

        top_cvs = [r for r in results if r.score > 80][:5]
        if top_cvs:
            async with websockets.connect('ws://localhost:8765') as websocket:
                message = f"Top CVs for job {job_id}: " + "; ".join(
                    [f"CV {r.cv_id} (Score: {r.score}, Email: {r.email}, Phone: {r.phone})" for r in top_cvs]
                )
                await websocket.send(message)
                logger.info(f"Sent notification to HR: {message}")

        return results
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)