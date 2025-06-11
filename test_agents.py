"""
Comprehensive test for AI Agent System with Orchestrator and Batch Processing
"""
import asyncio
import time
from agents import (
    OrchestratorAgent, BatchProcessingService, AgentMessage, MessageType,
    BatchProcessingRequest, JDAnalyzerAgent, CVAnalyzerAgent, MatchingAgent
)
from agent_integration_service import agent_integration_service

async def test_jd_analyzer():
    """Test JD Analyzer Agent"""
    print("\n🎯 Testing JD Analyzer Agent...")
    
    agent = JDAnalyzerAgent()
    
    # Sample job data
    job_data = {
        "id": "test_job_1",
        "title": "Senior Java Developer", 
        "description": """
        We are looking for a Senior Java Developer with 5+ years of experience.
        
        Required skills:
        - Java 8+, Spring Boot, MySQL
        - REST API development
        - Unit testing with JUnit
        
        Nice to have:
        - AWS experience
        - Docker/Kubernetes
        - Frontend skills (React)
        
        This is an urgent position for our fintech startup. Remote work available.
        Bachelor's degree in Computer Science preferred.
        """,
        "skillNames": ["Java", "Spring Boot", "MySQL", "REST API"]
    }
    
    # Create test message
    message = AgentMessage(
        agent_id="test",
        message_type=MessageType.ANALYSIS_REQUEST,
        data={"job_data": job_data}
    )
    
    try:
        # Process message
        result = await agent.process(message)
        
        if result.message_type == MessageType.ANALYSIS_RESULT:
            jd_analysis = result.data["jd_analysis"]
            print(f"✅ JD Analysis successful!")
            print(f"   Required skills: {jd_analysis['required_skills']}")
            print(f"   Nice-to-have skills: {jd_analysis['nice_to_have_skills']}")
            print(f"   Experience level: {jd_analysis['experience_level']}")
            print(f"   Urgency: {jd_analysis['urgency_level']}")
            print(f"   Confidence: {jd_analysis['confidence']}")
        else:
            print(f"❌ JD Analysis failed: {result.data}")
            
    except Exception as e:
        print(f"❌ JD Analysis error: {str(e)}")

async def test_cv_analyzer():
    """Test CV Analyzer Agent"""
    print("\n👤 Testing CV Analyzer Agent...")
    
    agent = CVAnalyzerAgent()
    
    # Sample CV data
    cv_data = {
        "id": "test_cv_1",
        "profile": "Experienced Java developer with 6 years in backend development",
        "skills": [
            {"name": "Java", "level": "Advanced"},
            {"name": "Spring Boot", "level": "Advanced"},
            {"name": "MySQL", "level": "Intermediate"}
        ],
        "experiences": [
            {
                "title": "Senior Java Developer",
                "company": "Tech Corp",
                "startDate": "2020-01",
                "endDate": "2024-12",
                "description": "Developed REST APIs using Spring Boot, managed MySQL databases"
            },
            {
                "title": "Java Developer", 
                "company": "StartupXYZ",
                "startDate": "2018-06",
                "endDate": "2019-12",
                "description": "Built web applications with Java and Spring framework"
            }
        ],
        "projects": [
            {
                "project": "E-commerce Platform",
                "description": "Built scalable e-commerce backend with Java, Spring Boot, MySQL",
                "startDate": "2023-01",
                "endDate": "2023-12"
            }
        ],
        "educations": [
            {
                "degree": "Bachelor's",
                "field": "Computer Science",
                "school": "Tech University",
                "graduationYear": "2018"
            }
        ],
        "certifications": ["Oracle Java Certification"],
        "languages": [{"language": "English", "level": "Fluent"}]
    }
    
    # Create test message
    message = AgentMessage(
        agent_id="test",
        message_type=MessageType.ANALYSIS_REQUEST,
        data={"cv_data": cv_data}
    )
    
    try:
        # Process message
        result = await agent.process(message)
        
        if result.message_type == MessageType.ANALYSIS_RESULT:
            cv_analysis = result.data["cv_analyses"][0]
            print(f"✅ CV Analysis successful!")
            print(f"   Skills: {cv_analysis['skills']}")
            print(f"   Experience years: {cv_analysis['experience_years']}")
            print(f"   Career level: {cv_analysis['career_level']}")
            print(f"   Strengths: {cv_analysis['strengths']}")
            print(f"   Growth potential: {cv_analysis['growth_potential']}")
            print(f"   Confidence: {cv_analysis['confidence']}")
        else:
            print(f"❌ CV Analysis failed: {result.data}")
            
    except Exception as e:
        print(f"❌ CV Analysis error: {str(e)}")

async def test_matching_agent():
    """Test Matching Agent"""
    print("\n🔗 Testing Matching Agent...")
    
    agent = MatchingAgent()
    
    # Sample JD and CV analysis results
    jd_analysis = {
        "job_id": "test_job_1",
        "required_skills": ["Java", "Spring Boot", "MySQL", "REST API"],
        "nice_to_have_skills": ["AWS", "Docker", "React"],
        "experience_level": "senior",
        "experience_years": 5,
        "soft_skills": ["communication", "teamwork"],
        "industry_context": "fintech startup",
        "urgency_level": "high",
        "cultural_requirements": ["startup mindset"],
        "education_requirements": ["Bachelor's degree"],
        "work_arrangement": "remote",
        "confidence": 0.9
    }
    
    cv_analysis = {
        "cv_id": "test_cv_1", 
        "skills": ["Java", "Spring Boot", "MySQL", "REST API", "JUnit"],
        "skill_levels": {"Java": "Advanced", "Spring Boot": "Advanced", "MySQL": "Intermediate"},
        "experience_years": 6,
        "career_level": "senior",
        "strengths": ["Strong backend development", "API design"],
        "weaknesses": ["Limited frontend experience"],
        "red_flags": [],
        "unique_selling_points": ["E-commerce domain expertise"],
        "career_progression": "Steady progression from developer to senior",
        "stability_score": 0.8,
        "growth_potential": "high",
        "education_background": ["Bachelor's in Computer Science"],
        "work_preference": "remote",
        "confidence": 0.85
    }
    
    # Create test message
    message = AgentMessage(
        agent_id="test",
        message_type=MessageType.MATCHING_REQUEST,
        data={
            "jd_analysis": jd_analysis,
            "cv_analysis": cv_analysis
        }
    )
    
    try:
        # Process message
        result = await agent.process(message)
        
        if result.message_type == MessageType.MATCHING_RESULT:
            matching = result.data["matching_results"][0]
            print(f"✅ Matching Analysis successful!")
            print(f"   Overall score: {matching['overall_score']}")
            print(f"   Skill match: {matching['skill_match_score']}")
            print(f"   Experience match: {matching['experience_match_score']}")
            print(f"   Matched skills: {matching['matched_skills']}")
            print(f"   Missing skills: {matching['missing_skills']}")
            print(f"   Recommended action: {matching['recommended_action']}")
            print(f"   Confidence: {matching['confidence']}")
        else:
            print(f"❌ Matching failed: {result.data}")
            
    except Exception as e:
        print(f"❌ Matching error: {str(e)}")

async def test_agent_health():
    """Test agent health checks"""
    print("\n🏥 Testing Agent Health Checks...")
    
    agents = [
        ("JD Analyzer", JDAnalyzerAgent()),
        ("CV Analyzer", CVAnalyzerAgent()), 
        ("Matching Agent", MatchingAgent())
    ]
    
    for name, agent in agents:
        try:
            health = await agent.health_check()
            status = health["status"]
            print(f"   {name}: {status}")
            if status == "unhealthy":
                print(f"     Error: {health.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   {name}: Failed - {str(e)}")

async def test_orchestrator_agent():
    """Test Orchestrator Agent with batch processing"""
    print("\n🎛️ Testing Orchestrator Agent...")
    
    orchestrator = OrchestratorAgent()
    
    # Test single match
    print("\n   Testing single match...")
    try:
        result = await orchestrator.single_match("test_job_1", "test_cv_1")
        print(f"✅ Single match successful! Score: {result.overall_score}")
    except Exception as e:
        print(f"❌ Single match failed: {str(e)}")
    
    # Test batch processing
    print("\n   Testing batch processing...")
    try:
        batch_result = await orchestrator.batch_process_job(
            job_id="test_job_1",
            cv_ids=["cv_1", "cv_2", "cv_3", "cv_4", "cv_5"],
            batch_size=3
        )
        print(f"✅ Batch processing successful!")
        print(f"   Processed: {batch_result.processed_cvs}/{batch_result.total_cvs}")
        print(f"   Processing time: {batch_result.processing_time:.2f}s")
        print(f"   Average score: {batch_result.summary.get('average_score', 'N/A')}")
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")

async def test_batch_processing_service():
    """Test Batch Processing Service"""
    print("\n🔄 Testing Batch Processing Service...")
    
    service = BatchProcessingService(max_concurrent_jobs=2, default_batch_size=10)
    
    # Test job submission
    print("\n   Submitting batch job...")
    try:
        job_ids = ["cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "cv_6"]
        batch_job_id = await service.submit_batch_job(
            job_id="test_job_batch",
            cv_ids=job_ids,
            batch_size=3,
            priority="high"
        )
        print(f"✅ Batch job submitted: {batch_job_id}")
        
        # Monitor job progress
        max_wait = 30  # 30 seconds max
        waited = 0
        while waited < max_wait:
            status = await service.get_job_status(batch_job_id)
            if status:
                print(f"   Job status: {status['status']} - Progress: {status['progress']}%")
                
                if status['status'] == 'completed':
                    result = await service.get_job_result(batch_job_id)
                    if result:
                        print(f"✅ Batch job completed successfully!")
                        print(f"   Results: {len(result.results)} matches")
                        print(f"   Processing time: {result.processing_time:.2f}s")
                    break
                elif status['status'] == 'failed':
                    print(f"❌ Batch job failed: {status.get('error', 'Unknown error')}")
                    break
            
            await asyncio.sleep(2)
            waited += 2
        
        if waited >= max_wait:
            print("⏰ Batch job timed out")
            
    except Exception as e:
        print(f"❌ Batch processing service test failed: {str(e)}")
    
    # Test service metrics
    print("\n   Getting service metrics...")
    try:
        metrics = await service.get_service_metrics()
        print(f"✅ Service metrics retrieved:")
        print(f"   Active jobs: {metrics['active_jobs']}")
        print(f"   Completed jobs: {metrics['completed_jobs']}")
        print(f"   LLM calls saved: {metrics['total_llm_calls_saved']}")
    except Exception as e:
        print(f"❌ Failed to get metrics: {str(e)}")

async def test_agent_integration_service():
    """Test Agent Integration Service"""
    print("\n🔗 Testing Agent Integration Service...")
    
    from models import CvInput, JdInput
    
    # Create test data
    test_cvs = [
        CvInput(
            cv_id="integration_cv_1",
            skills=["Python", "FastAPI", "PostgreSQL"],
            experience="3 years backend development",
            education="Bachelor's in Computer Science",
            text="Backend developer with API experience",
            email="dev1@test.com",
            phone="+1234567890"
        ),
        CvInput(
            cv_id="integration_cv_2", 
            skills=["Java", "Spring Boot", "MySQL"],
            experience="5 years enterprise development",
            education="Master's in Software Engineering",
            text="Senior Java developer with enterprise experience",
            email="dev2@test.com",
            phone="+1234567891"
        )
    ]
    
    test_jd = JdInput(
        required_skills=["Python", "API Development", "Database"],
        required_experience="2+ years backend development",
        text="Looking for backend developer with Python and API experience"
    )
    
    # Test small batch (immediate processing)
    print("\n   Testing small batch processing...")
    try:
        start_time = time.time()
        results = await agent_integration_service.match_cvs_with_agents(
            cvs=test_cvs,
            jd=test_jd,
            use_batch_processing=False
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Small batch processing successful!")
        print(f"   Results: {len(results)} matches")
        print(f"   Processing time: {processing_time:.2f}s")
        for result in results:
            print(f"   CV {result.cv_id}: {result.score:.1f} - {result.recommended_action}")
    except Exception as e:
        print(f"❌ Small batch processing failed: {str(e)}")
    
    # Test service health
    print("\n   Testing service health...")
    try:
        health = await agent_integration_service.get_service_health()
        print(f"✅ Service health: {health['status']}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

async def test_performance_optimization():
    """Test performance optimization comparison"""
    print("\n⚡ Testing Performance Optimization...")
    
    # Simulate different batch sizes
    cv_counts = [10, 50, 100]
    batch_sizes = [5, 10, 20]
    
    print("\n   LLM Call Optimization Analysis:")
    print("   CV Count | Batch Size | Original Calls | Optimized Calls | Savings")
    print("   ---------|------------|----------------|-----------------|--------")
    
    for cv_count in cv_counts:
        for batch_size in batch_sizes:
            original_calls = 1 + 2 * cv_count
            optimized_calls = 1 + 2 * ((cv_count + batch_size - 1) // batch_size)
            savings = original_calls - optimized_calls
            savings_percent = (savings / original_calls) * 100
            
            print(f"   {cv_count:8d} | {batch_size:10d} | {original_calls:14d} | {optimized_calls:15d} | {savings:3d} ({savings_percent:4.1f}%)")

async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive AI Agent System Tests...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test individual agents
    await test_jd_analyzer()
    await test_cv_analyzer()
    await test_matching_agent()
    
    # Test orchestrator
    await test_orchestrator_agent()
    
    # Test batch processing service
    await test_batch_processing_service()
    
    # Test integration service
    await test_agent_integration_service()
    
    # Test performance optimization
    await test_performance_optimization()
    
    total_time = time.time() - start_time
    print(f"\n✅ All tests completed in {total_time:.2f} seconds!")
    print("=" * 60)

if __name__ == "__main__":
    # Run comprehensive tests
    asyncio.run(run_comprehensive_tests())
