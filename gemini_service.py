import httpx
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
import os
import logging

# Initialize NLTK data
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt data")
        nltk.download('punkt', quiet=True)

init_nltk()

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
API_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")

logger = logging.getLogger(__name__)

class ExtractedData(BaseModel):
    skills: List[str] = []
    experience: str = ""
    education: str = ""
    certifications: List[str] = []
    projects: Optional[List[Dict[str, str]]] = None
    languages: Optional[List[Dict[str, str]]] = None
    years_experience: Optional[float] = None

async def extract_data(text: str, type: str) -> ExtractedData:
    if not text.strip():
        logger.warning(f"Empty {type} text provided, returning default ExtractedData")
        return ExtractedData(
            skills=[],
            experience="",
            education="",
            certifications=[],
            projects=[],
            languages=[],
            years_experience=None
        )
    
    prompt = (
        f"Extract the following from the {type} text in JSON format: "
        "skills (list of strings, include technical skills only, exclude language proficiency), "
        "experience (summary as a string, include years and roles), "
        "education (summary as a string, include degrees and fields), "
        "certifications (list of strings), "
        "projects (list of objects with project name and description, return empty list if none), "
        "languages (list of objects with language and proficiency level, return empty list if none). "
        "Return ONLY the JSON object, no explanatory text, markdown, or additional content."
    )
    
    request_body = {
        "contents": [{"parts": [{"text": f"{prompt}\n\nText: {text}"}]}]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_ENDPOINT}?key={API_KEY}",
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            
            response_data = response.json()
            logger.debug(f"Gemini response for {type} extraction: {response_data}")
            
            if not response_data.get("candidates"):
                logger.error("No candidates found in Gemini response")
                raise ValueError("Empty candidates in Gemini response")
                
            candidate = response_data["candidates"][0]
            if not candidate.get("content") or not candidate["content"].get("parts"):
                logger.error("No content or parts found in Gemini response")
                raise ValueError("Empty content in Gemini response")
                
            text_content = candidate["content"]["parts"][0]["text"].strip()
            json_content = text_content.replace("```json\n", "").replace("\n```", "").strip()
            
            try:
                data = json.loads(json_content)
                years = None
                tokens = word_tokenize(text.lower())
                for i, token in enumerate(tokens):
                    if token in ["year", "years", "yr", "yrs"]:
                        if i > 0 and re.match(r'^\d+\.?\d*$', tokens[i-1]):
                            years = float(tokens[i-1])
                            break
                data["years_experience"] = years
                data.setdefault("projects", [])
                data.setdefault("languages", [])
                for lang in data.get("languages", []):
                    lang["level"] = lang.get("level", "") if lang.get("level") is not None else ""
                return ExtractedData(**data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Gemini response: {json_content}")
                raise ValueError(f"Invalid JSON in Gemini response: {str(e)}")
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Gemini API: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

async def calculate_relevance(cv_data: ExtractedData, jd_data: ExtractedData) -> Tuple[float, str]:
    prompt = (
        "You are an expert in evaluating CVs against Job Descriptions (JDs) for recruitment. Given a CV and a JD, perform a comprehensive evaluation to determine how well the CV matches the JD. Follow these steps:\n\n"
        "1. **Extract Requirements from JD**:\n"
        "   - Identify mandatory skills (technical skills like Java, Spring Boot, etc., listed in `skillNames` or mentioned in `description`).\n"
        "   - Identify required experience (years, roles, or specific domains mentioned in `description` or `experienceYear`).\n"
        "   - Identify required education (degrees, fields mentioned in `description`).\n"
        "   - Identify required certifications (if any).\n"
        "   - Identify required languages and proficiency levels (e.g., 'fluent in English' from `description`).\n"
        "   - Identify required project experience (specific project types or domains mentioned in `description`).\n\n"
        "2. **Extract Information from CV**:\n"
        "   - Identify skills from `skills` (technical and soft skills).\n"
        "   - Identify experience from `experience` (summary of years, roles, domains).\n"
        "   - Identify education from `education` (degrees, fields).\n"
        "   - Identify certifications from `certificates`.\n"
        "   - Identify languages and proficiency levels from `languages`.\n"
        "   - Identify project experience from `projects` (project names, descriptions).\n\n"
        "3. **Evaluate Match with Weighted Scoring**:\n"
        "   - Assign weights to each category:\n"
        "     - Skills: 40% (prioritize technical skills, partial match for related skills)\n"
        "     - Experience: 30% (consider years, roles, and domain relevance)\n"
        "     - Education: 10% (full match for exact degrees/fields, partial for related)\n"
        "     - Certifications: 5% (full match for listed certifications)\n"
        "     - Languages: 10% (full match for required languages with sufficient proficiency)\n"
        "     - Projects: 5% (match based on relevance to JD requirements)\n"
        "   - For each category:\n"
        "     - Calculate a match score (0-100) based on how well the CV meets the JD requirements.\n"
        "     - Consider semantic similarity (e.g., 'API development' in JD matches 'Built RESTful APIs' in CV projects).\n"
        "     - Account for partial matches (e.g., 2 years experience vs 3 years required gives partial score).\n"
        "   - Compute the final score as a weighted sum of category scores.\n\n"
        "4. **Provide Detailed Explanation**:\n"
        "   - For each category, explain the match score, including matched items, missing items, and partial matches.\n"
        "   - Highlight any inferred matches (e.g., skills inferred from projects or experience).\n\n"
        "**Input**:\n"
        "- **JD**:\n"
        "  - skillNames: {jd_skillNames}\n"
        "  - description: {jd_description}\n"
        "  - experienceYear: {jd_experienceYear}\n"
        "- **CV**:\n"
        "  - skills: {cv_skills}\n"
        "  - experience: {cv_experience}\n"
        "  - education: {cv_education}\n"
        "  - certificates: {cv_certificates}\n"
        "  - languages: {cv_languages}\n"
        "  - projects: {cv_projects}\n\n"
        "**Output**:\n"
        "Return a JSON object with:\n"
        "- score: float (0-100, overall match score)\n"
        "- explanation: string (detailed explanation of the match for each category)\n\n"
        "Example Output:\n"
        "```json\n"
        "{\"score\": 92.5, \"explanation\": \"Skills: Matched 4/5 mandatory skills (80%); Experience: CV 3 years vs JD 3 years (100% match); Education: Matched BS in Computer Science (100%); Certifications: Matched 0/0 certifications (100%); Languages: Matched English (Fluent) (100%); Projects: Matched e-learning project (100%)\"}\n"
        "```\n\n"
        "**Notes**:\n"
        "- Be strict with mandatory skills listed in `skillNames`.\n"
        "- Infer skills from CV `projects` or `experience` if not listed in `skills`.\n"
        "- For languages, match proficiency levels (e.g., 'Fluent' or 'Advanced' satisfies 'fluent in English').\n"
        "- Return only the JSON object, no additional text or markdown."
    )
    
    cv_text = json.dumps({
        "skills": cv_data.skills,
        "experience": cv_data.experience,
        "education": cv_data.education,
        "certificates": cv_data.certifications,
        "languages": cv_data.languages,
        "projects": cv_data.projects
    })
    jd_text = json.dumps({
        "skillNames": jd_data.skills,
        "description": jd_data.experience,
        "experienceYear": str(jd_data.years_experience or "")
    })
    
    request_body = {
        "contents": [{"parts": [{"text": prompt.replace("{jd_skillNames}", json.dumps(jd_data.skills))
                                       .replace("{jd_description}", jd_data.experience)
                                       .replace("{jd_experienceYear}", str(jd_data.years_experience or ""))
                                       .replace("{cv_skills}", json.dumps(cv_data.skills))
                                       .replace("{cv_experience}", json.dumps(cv_data.experience))
                                       .replace("{cv_education}", json.dumps(cv_data.education))
                                       .replace("{cv_certificates}", json.dumps(cv_data.certifications))
                                       .replace("{cv_languages}", json.dumps(cv_data.languages))
                                       .replace("{cv_projects}", json.dumps(cv_data.projects))}]}]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_ENDPOINT}?key={API_KEY}",
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            
            response_data = response.json()
            logger.debug(f"Gemini response for relevance: {response_data}")
            
            if not response_data.get("candidates"):
                logger.error("No candidates found in Gemini response for relevance")
                raise ValueError("Empty candidates in Gemini response")
                
            candidate = response_data["candidates"][0]
            if not candidate.get("content") or not candidate["content"].get("parts"):
                logger.error("No content or parts found in Gemini response for relevance")
                raise ValueError("Empty content in Gemini response")
                
            text_content = candidate["content"]["parts"][0]["text"].strip()
            json_content = text_content.replace("```json\n", "").replace("\n```", "").strip()
            
            try:
                result = json.loads(json_content)
                score = result.get("score", 0.0)
                explanation = result.get("explanation", "")
                return score, explanation
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Gemini response: {json_content}")
                raise ValueError(f"Invalid JSON in Gemini response: {str(e)}")
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Gemini API: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

async def normalize_data(data: dict, type: str) -> ExtractedData:
    try:
        if type == "JD":
            text = data.get("text", "")
            skills = data.get("required_skills", [])
            experience = data.get("required_experience", "")
            education = data.get("required_education", "")
            certifications = data.get("required_certifications", [])
            extracted = await extract_data(text, type)
            languages = extracted.languages or []
            if "fluent in English".lower() in text.lower():
                languages.append({"language": "English", "level": "Fluent"})
            return ExtractedData(
                skills=skills or extracted.skills,
                experience=experience or extracted.experience,
                education=education or extracted.education,
                certifications=certifications or extracted.certifications,
                projects=extracted.projects or [],
                languages=languages,
                years_experience=extracted.years_experience
            )
        else:
            text = data.get("text", "")
            extracted = await extract_data(text, type)
            languages = [
                {"language": l["language"], "level": l["level"] if l["level"] is not None else ""}
                for l in data.get("languages", extracted.languages or [])
            ]
            return ExtractedData(
                skills=data.get("skills", extracted.skills),
                experience=data.get("experience", extracted.experience),
                education=data.get("education", extracted.education),
                certifications=data.get("certifications", extracted.certifications),
                projects=data.get("projects", extracted.projects or []),
                languages=languages,
                years_experience=extracted.years_experience
            )
    except Exception as e:
        logger.error(f"Error normalizing {type} data: {str(e)}")
        raise ValueError(f"Invalid {type} data: {str(e)}")