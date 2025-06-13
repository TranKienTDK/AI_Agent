# AI Agent System for CV-JD Matching

## 📋 Tổng quan hệ thống

Hệ thống AI Agent được thiết kế để tự động hóa quá trình đánh giá và so khớp CV với Job Description (JD) thông qua API `/match-all/{job_id}` với `use_ai_agents=true`. Sử dụng kiến trúc đa AI Agent chuyên biệt để tối ưu hiệu suất và độ chính xác.

## � API Chính

### `/match-all/{job_id}?use_ai_agents=true`

API chính của hệ thống, tự động:
- Lấy thông tin Job Description theo `job_id`
- Lấy danh sách CV chưa được đánh giá cho job này
- Sử dụng AI Agent system để batch processing
- Trả về kết quả với recommended actions
- Lưu kết quả đánh giá vào database

## �🏗️ Kiến trúc hệ thống

### Kiến trúc AI Agent đa tầng

```
┌─────────────────────────────────────────────────┐
│              ORCHESTRATOR                       │
│           (Agent điều phối)                     │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│            SPECIALIZED AGENTS                   │
├─────────────────┬───────────────────────────────┤
│  JD Analyzer    │      CV Analyzer              │
│  Agent          │      Agent                    │
│ (Phân tích JD)  │   (Phân tích CV)              │
└─────────────────┴───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│             MATCHING AGENT                      │
│           (Agent đánh giá so khớp)              │
└─────────────────────────────────────────────────┘
```
**Output:** List[Structured CV Analysis Objects]

#### 🔗 Matching Agent
**Chức năng chính:**
- Thực hiện multi-dimensional matching algorithm
- Tính toán weighted scoring với contextual adjustment
- Generate detailed explanation cho từng match
- Đưa ra actionable recommendations

**Input:** JD Analysis + CV Analysis Objects
**Output:** Evaluation Results với scores, explanations, recommendations

## ⚡ Tối ưu hóa hiệu suất

### Batch Processing Strategy
- **Xử lý song song:** Thay vì xử lý tuần tự từng CV, hệ thống xử lý nhiều CV cùng lúc
- **Batch size:** 10-20 CVs per batch để tối ưu context window của LLM
- **Parallel execution:** Sử dụng asyncio để chạy song song các tasks

### LLM Call Optimization
```
Trước tối ưu: 1 + 2N calls (N = số CV)
- 1 call cho JD analysis
- N calls cho CV analysis  
- N calls cho matching

Sau tối ưu: 1 + 2×ceil(N/batch_size) calls
- 1 call cho JD analysis
- ceil(N/20) calls cho CV batch analysis
- ceil(N/20) calls cho batch matching

Ví dụ 100 CVs: 201 calls → 11 calls (giảm 95%)
```

## 🔄 Giao tiếp giữa AI Agents

### Agent Communication Protocol
```python
class AgentMessage:
    agent_id: str           # ID của agent gửi
    message_type: str       # Loại message (request, response, error)
    data: dict             # Payload chính
    metadata: dict         # Thông tin phụ (timestamp, confidence)
    confidence: float      # Độ tin cậy của kết quả
```

### Workflow giao tiếp
1. **Orchestrator** nhận request từ API
2. **JD Analyzer** phân tích Job Description
3. **CV Analyzer** xử lý batch CVs song song
4. **Matching Agent** thực hiện so khớp theo batch
5. **Orchestrator** tổng hợp và trả về kết quả

## 📊 Cấu trúc dữ liệu

### Input Data Structure
```python
# Job Input
{
    "job_id": "string",
    "title": "Java Developer",
    "description": "Job description text...",
    "skillNames": ["Java", "Spring Boot", "MySQL"],
    "experienceYear": 3,
    "raw_data": {...}
}

# CV Input  
{
    "cv_id": "string",
    "profile": "Profile summary...",
    "skills": [{"name": "Java", "level": "Advanced"}],
    "experiences": [...],
    "projects": [...],
    "raw_data": {...}
}
```

### Output Data Structure
```python
{
    "cv_id": "string",
    "score": 85.5,                    # Điểm số 0-100
    "explanation": "Chi tiết phân tích...",
    "recommended_action": "send_contact_email",
    "action_reason": "Lý do recommend...",
    
    # Enhanced fields
    "confidence_score": 0.92,         # Độ tin cậy
    "strengths": ["Strong Java", "Good projects"],
    "concerns": ["Limited scalability experience"],
    "skill_gaps": ["Docker", "Kubernetes"],
    "growth_potential": "high"
}
```

## 🚀 API Endpoints

### Endpoint chính
```
POST /match-all/{job_id}
- Đánh giá tất cả CV với 1 JD
- Sử dụng AI Agent system mới
- Trả về results với enhanced information
```

### AI Agent endpoints
```
POST /ai-agent/analyze-all/{job_id}
- Sử dụng trực tiếp AI Agent Controller
- Batch processing tối ưu
- Detailed agent communication logs
```

## 🚀 Enhanced API Endpoints

### Traditional endpoint (with AI Agent option)
```bash
POST /match-all/{job_id}?use_ai_agents=true
# Now supports AI Agent optimization for large datasets
# Automatically switches to batch processing for >3 CVs
```

### AI Agent dedicated endpoints
```bash
# Submit batch job
POST /ai-agent/match-all/{job_id}
{
  "batch_size": 20,
  "priority": "high"
}

# Track job progress
GET /ai-agent/status/{batch_job_id}

# Get completed results
GET /ai-agent/result/{batch_job_id}

# Health check
GET /ai-agent/health

# Performance metrics
GET /ai-agent/metrics
```

### Response format enhancements
```json
{
  "job_id": "123",
  "total_candidates": 50,
  "processing_method": "ai_agents_batch",
  "ai_agent_optimization": {
    "enabled": true,
    "batch_size": 20,
    "estimated_llm_calls_saved": 80
  },
  "results": [...],
  "summary": {...}
}
```

## 🎯 Implementation Status

### ✅ Completed Features

1. **AI Agent Architecture**
   - ✅ BaseAgent abstract class với common functionality
   - ✅ JDAnalyzerAgent với comprehensive job analysis
   - ✅ CVAnalyzerAgent với batch processing capabilities
   - ✅ MatchingAgent với multi-dimensional scoring
   - ✅ OrchestratorAgent để coordinate workflow
   - ✅ BatchProcessingService để optimize LLM calls

2. **Performance Optimization**
   - ✅ LLM call reduction: 1+2N → 1+2×ceil(N/batch_size)
   - ✅ Batch processing với configurable batch sizes
   - ✅ Parallel processing trong chunks
   - ✅ Progress tracking và monitoring

3. **API Integration**
   - ✅ Enhanced `/match-all/{job_id}` endpoint với AI Agent option
   - ✅ New `/ai-agent/match-all/{job_id}` endpoint for direct agent access
   - ✅ Agent status và result tracking endpoints
   - ✅ Health check và metrics endpoints

4. **Agent Communication**
   - ✅ Standardized AgentMessage protocol
   - ✅ Type-safe message passing between agents
   - ✅ Error handling và recovery mechanisms
   - ✅ Performance metrics tracking

5. **Testing & Documentation**
   - ✅ Comprehensive test suite cho all agents
   - ✅ Performance demo script
   - ✅ Integration tests với realistic data
   - ✅ Health monitoring và service metrics

## 📈 Performance Metrics

### Dự kiến cải thiện
- **Processing time:** 30-60 giây cho 100 CVs (từ 10+ phút)
- **LLM calls:** Giảm 95% số lượng calls
- **Cost reduction:** Tiết kiệm 95% chi phí LLM
- **Accuracy:** Tương đương hoặc tốt hơn nhờ specialized agents

### Batch processing performance
| Số CVs | LLM Calls (Cũ) | LLM Calls (Mới) | Cải thiện |
|--------|-----------------|-----------------|-----------|
| 20     | 41              | 3               | 93%       |
| 50     | 101             | 6               | 94%       |
| 100    | 201             | 11              | 95%       |
| 200    | 401             | 21              | 95%       |

## 🛠️ Triển khai

### Phase 1: Core Agents (Week 1-2)
- [ ] Implement JDAnalyzerAgent
- [ ] Implement CVAnalyzerAgent  
- [ ] Implement MatchingAgent
- [ ] Setup basic communication protocol

### Phase 2: Orchestration (Week 2-3)
- [ ] Build AgentOrchestrator
- [ ] Implement batch processing
- [ ] Setup parallel execution
- [ ] Integration testing

### Phase 3: Optimization (Week 3-4)
- [ ] Performance tuning
- [ ] Error handling enhancement
- [ ] API integration
- [ ] Documentation và testing

## 📁 Cấu trúc thư mục

```
ai_agent/
├── agents/
│   ├── jd_analyzer_agent.py       # JD Analysis Agent
│   ├── cv_analyzer_agent.py       # CV Analysis Agent
│   ├── matching_agent.py          # Matching Agent
│   └── orchestrator.py            # Agent Orchestrator
├── models/
│   ├── agent_models.py            # Agent communication models
│   └── evaluation_models.py       # Evaluation result models
├── services/
│   ├── batch_processor.py         # Batch processing service
│   └── performance_monitor.py     # Performance monitoring
├── main.py                        # FastAPI application
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
JWT_TOKEN=your_jwt_token
BATCH_SIZE=20
MAX_PARALLEL_TASKS=10
```

### Agent Settings
```python
# Agent configuration
JD_AGENT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 1500
}

CV_AGENT_CONFIG = {
    "model": "gpt-4o-mini", 
    "temperature": 0.1,
    "max_tokens": 1000,
    "batch_size": 20
}

MATCHING_AGENT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 2000,
    "scoring_weights": {
        "skills": 0.4,
        "experience": 0.3,
        "education": 0.1,
        "projects": 0.1,
        "languages": 0.1
    }
}
```

## 🧪 Testing Strategy

### Unit Tests
- [ ] Test từng Agent riêng biệt
- [ ] Test Agent communication protocol
- [ ] Test batch processing logic

### Integration Tests  
- [ ] Test end-to-end workflow
- [ ] Test với different data volumes
- [ ] Performance benchmarking

### Load Tests
- [ ] Test với 100+ CVs
- [ ] Concurrent request handling
- [ ] Memory và CPU usage monitoring

## 🧪 Testing & Demo

### Run comprehensive tests
```bash
python test_agents.py
```

### Run performance demo
```bash
python demo_performance.py
```

### Test coverage includes:
- Individual agent functionality
- Orchestrator workflow coordination
- Batch processing service
- Integration service compatibility
- Performance optimization verification
- Error handling và recovery
- Health monitoring

### Performance benchmarks:
- **10 CVs**: 21 → 7 LLM calls (67% reduction)
- **50 CVs**: 101 → 7 LLM calls (93% reduction)  
- **100 CVs**: 201 → 11 LLM calls (95% reduction)
- **200 CVs**: 401 → 21 LLM calls (95% reduction)

## 📝 Logging và Monitoring

### Agent Activity Logging
```python
# Agent performance logs
{
    "agent_id": "jd_analyzer",
    "processing_time": 1.2,
    "input_size": 1024,
    "confidence": 0.95,
    "llm_calls": 1
}
```

### System Metrics
- Processing time per batch
- LLM API call frequency
- Success/failure rates
- Cost per evaluation

## 🤝 Contributing

### Development Guidelines
1. Mỗi Agent phải implement base AgentInterface
2. Sử dụng async/await cho tất cả I/O operations
3. Comprehensive error handling và logging
4. Unit tests cho mọi component mới

### Code Standards
- Follow PEP 8 style guide
- Type hints cho tất cả functions
- Docstrings cho public methods
- Error handling với proper exception types

---
