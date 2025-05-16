import asyncio
import websockets
import json
import csv
import logging
from sentence_transformers import SentenceTransformer, util
from gemini_service import normalize_data, calculate_relevance
from models import CvInput, JdInput, CvMatchResult
from typing import List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

async def notify_hr(message: str):
    async with websockets.connect('ws://localhost:8765') as websocket:
        await websocket.send(message)
        logger.info(f"Sent notification to HR: {message}")

def save_evaluation_result(cv_id: str, score: float, feedback: str = None):
    file_exists = os.path.isfile('evaluations.csv')
    with open('evaluations.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['cv_id', 'score', 'feedback'])
        writer.writerow([cv_id, score, feedback])
    logger.info(f"Saved evaluation for CV {cv_id}: score={score}, feedback={feedback}")

async def match_cvs_with_agent(cvs: List[CvInput], jd: JdInput) -> List[CvMatchResult]:
    try:
        if not cvs:
            logger.error("Empty CV list provided")
            raise ValueError("Empty CV list provided")

        logger.info("Normalizing JD data")
        jd_data = await normalize_data(jd.dict(), "JD")
        logger.info(f"JD data: {jd_data}")

        results = []
        for cv in cvs:
            try:
                logger.info(f"Processing CV: {cv.cv_id}")
                cv_data = await normalize_data(cv.dict(), "CV")
                logger.info(f"CV {cv.cv_id} data: {cv_data}")

                score, explanation = await calculate_relevance(cv_data, jd_data)
                results.append(CvMatchResult(
                    cv_id=cv.cv_id,
                    score=score,
                    explanation=explanation,
                    email=cv.email,
                    phone=cv.phone
                ))
                logger.info(f"CV {cv.cv_id} score: {score}, explanation: {explanation}")

                save_evaluation_result(cv_id=cv.cv_id, score=score)

                if score > 80:
                    await notify_hr(f"High-scoring CV detected: {cv.cv_id} with score {score}")

            except Exception as e:
                logger.error(f"Error processing CV {cv.cv_id}: {str(e)}")
                results.append(CvMatchResult(
                    cv_id=cv.cv_id,
                    score=0.0,
                    explanation=f"Error: {str(e)}",
                    email=cv.email,
                    phone=cv.phone
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Returning {len(results)} results")
        return results

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise

async def websocket_server():
    async def handler(websocket, path):
        async for message in websocket:
            logger.info(f"Received message from HR: {message}")
    server = await websockets.serve(handler, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(websocket_server())