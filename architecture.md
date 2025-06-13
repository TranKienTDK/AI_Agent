# AI Agent System Architecture

## 🏗️ Kiến trúc tổng thể hệ thống

```mermaid
graph TB
    subgraph "Frontend (React/Vue)"
        FE[Frontend Application]
        UI[UI Components]
        FE --> UI
    end

    subgraph "AI Agent System (FastAPI - Port 8000)"
        API["/match-all/{job_id}"]
        MAIN[main.py]
        AIS[agent_integration_service.py]
        
        subgraph "AI Agents"
            ORCH[OrchestratorAgent]
            JDA[JDAnalyzerAgent]
            CVA[CVAnalyzerAgent]
            MA[MatchingAgent]
            BPS[BatchProcessingService]
        end
        
        subgraph "Support Services"
            OAI[openai_service.py]
            FT[function_tools.py]
            MODELS[models.py]
        end
    end

    subgraph "Backend API (Spring Boot - Port 8080)"
        BE[Backend Server]
        JOBAPI["/api/v1/job/{id}"]
        CVAPI["/api/v1/cv/all"]
        EVALAPI["/api/v1/evaluations"]
        DB[(Database)]
    end

    subgraph "External Services"
        OPENAI[OpenAI API]
    end

    %% Frontend to AI Agent
    FE -->|"POST /match-all/{job_id}"| API
    API --> MAIN
    MAIN --> AIS

    %% AI Agent Internal Flow
    AIS --> ORCH
    ORCH --> JDA
    ORCH --> CVA
    ORCH --> MA
    AIS --> BPS

    %% AI Agent to Backend
    MAIN -->|"GET /api/v1/job/{id}"| JOBAPI
    MAIN -->|"GET /api/v1/cv/all?jobId={id}"| CVAPI
    MAIN -->|"POST /api/v1/evaluations"| EVALAPI
    FT -->|"API calls"| BE

    %% Backend to Database
    JOBAPI --> DB
    CVAPI --> DB
    EVALAPI --> DB

    %% AI Agents to OpenAI
    JDA -->|"via openai_service"| OAI
    CVA -->|"via openai_service"| OAI
    MA -->|"via openai_service"| OAI
    OAI --> OPENAI

    %% Support connections
    ORCH -.-> MODELS
    AIS -.-> MODELS
    JDA -.-> OAI
    CVA -.-> OAI
    MA -.-> OAI

    style FE fill:#e1f5fe
    style API fill:#fff3e0
    style ORCH fill:#f3e5f5
    style JDA fill:#e8f5e8
    style CVA fill:#e8f5e8
    style MA fill:#e8f5e8
    style BE fill:#fce4ec
    style OPENAI fill:#fff8e1
```

## 🔄 Flow chi tiết khi gọi API `/match-all/{job_id}`

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant AI as AI Agent API
    participant AIS as AgentIntegrationService
    participant ORCH as OrchestratorAgent
    participant JDA as JDAnalyzerAgent
    participant CVA as CVAnalyzerAgent
    participant MA as MatchingAgent
    participant BE as Backend API
    participant OPENAI as OpenAI API

    Note over FE,OPENAI: User clicks "Evaluate CVs" button

    FE->>AI: POST /match-all/{job_id}?use_ai_agents=true
    AI->>BE: GET /api/v1/job/{job_id}
    BE-->>AI: Job data (title, description, skills)
    
    AI->>BE: GET /api/v1/cv/all?jobId={job_id}
    BE-->>AI: List of unevaluated CVs
    
    Note over AI: Check if use_ai_agents=true & len(cvs) > 3
    AI->>AIS: match_cvs_with_agents(cvs, jd, batch_processing=true)
    
    alt Batch Processing (>3 CVs)
        AIS->>ORCH: Submit batch job
        
        Note over ORCH,OPENAI: Step 1: Analyze JD (1 LLM call)
        ORCH->>JDA: Analyze job description
        JDA->>OPENAI: Analyze JD requirements
        OPENAI-->>JDA: Structured JD analysis
        JDA-->>ORCH: JDAnalysisResult
        
        Note over ORCH,OPENAI: Step 2: Analyze CVs in batches (ceil(N/20) LLM calls)
        ORCH->>CVA: Analyze CV batch (20 CVs)
        CVA->>OPENAI: Analyze CV batch
        OPENAI-->>CVA: CV analysis results
        CVA-->>ORCH: List[CVAnalysisResult]
        
        Note over ORCH,OPENAI: Step 3: Match in batches (ceil(N/20) LLM calls)
        ORCH->>MA: Match CVs with JD
        MA->>OPENAI: Perform matching analysis
        OPENAI-->>MA: Matching results with scores
        MA-->>ORCH: List[MatchingResult]
        
        ORCH-->>AIS: BatchProcessingResult
        AIS-->>AI: List[CvMatchResult]
        
    else Small Batch (≤3 CVs)
        Note over AIS: Use immediate processing
        AIS->>AI: Direct processing without queue
    end
    
    Note over AI,BE: Save results to database
    loop For each result
        AI->>BE: POST /api/v1/evaluations
        BE-->>AI: Saved successfully
    end
    
    AI-->>FE: {results, summary, ai_agent_optimization}
    
    Note over FE: Display results with recommended actions
```

## 📊 Tối ưu hóa LLM calls

```mermaid
graph LR
    subgraph "Traditional Method"
        T1[1 JD Analysis] --> T2[CV1 Analysis]
        T2 --> T3[CV1 Matching]
        T3 --> T4[CV2 Analysis]
        T4 --> T5[CV2 Matching]
        T5 --> T6[... CVN Analysis]
        T6 --> T7[CVN Matching]
        T7 --> TR[Total: 1 + 2N calls]
    end
    
    subgraph "AI Agent Batch Method"
        A1[1 JD Analysis] --> A2[Batch CV Analysis<br/>ceil(N/20) calls]
        A2 --> A3[Batch Matching<br/>ceil(N/20) calls]
        A3 --> AR[Total: 1 + 2×ceil(N/20) calls]
    end
    
    TR -.->|"95% reduction<br/>for 100 CVs"| AR
    
    style TR fill:#ffcdd2
    style AR fill:#c8e6c9
```

## 🎯 API Response Structure

```json
{
  "job_id": "job_123",
  "total_candidates": 50,
  "results": [
    {
      "cv_id": "cv_456",
      "score": 85.5,
      "explanation": "Strong technical match...",
      "recommended_action": "send_contact_email",
      "action_reason": "Ứng viên rất tiềm năng..."
    }
  ],
  "summary": {
    "send_contact_email": 12,
    "save_cv": 25,
    "no_recommendation": 13
  },
  "processing_method": "ai_agents_batch",
  "ai_agent_optimization": {
    "enabled": true,
    "batch_size": 20,
    "estimated_llm_calls_saved": 87
  }
}
```

## 🔧 Key Components

### 1. **Frontend Integration**
- Gọi API `/match-all/{job_id}` với parameter `use_ai_agents=true`
- Nhận kết quả với recommended actions
- Hiển thị summary và optimization metrics

### 2. **AI Agent System**
- **OrchestratorAgent**: Điều phối workflow
- **JDAnalyzerAgent**: Phân tích job description  
- **CVAnalyzerAgent**: Phân tích CVs theo batch
- **MatchingAgent**: Thực hiện matching với scoring

### 3. **Backend Integration**
- Cung cấp job data và CV data
- Lưu trữ kết quả evaluation
- Filter CVs chưa được đánh giá

### 4. **Performance Optimization**
- **Traditional**: 1 + 2N LLM calls (N = số CVs)
- **AI Agent Batch**: 1 + 2×ceil(N/batch_size) calls
- **Example**: 100 CVs: 201 calls → 11 calls (95% reduction)
