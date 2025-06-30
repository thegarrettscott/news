# News Aggregation API

A FastAPI-based news aggregation and summarization service that provides intelligent briefings on recent news topics using AI-powered research and content extraction.

## 🏗️ Architecture Overview

### Core Components

The news system is built around a single FastAPI application (`main.py`) with the following key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    News API Layout                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📡 API Endpoint: GET /news                                 │
│  ├── Query Parameters:                                      │
│  │   ├── topic (required)                                   │
│  │   ├── user (optional - enables async processing)        │
│  │   ├── date_range (default: "past 2 days")              │
│  │   ├── effort (low/medium/high)                          │
│  │   ├── debug (boolean)                                   │
│  │   ├── previous_summary (string)                         │
│  │   ├── max_steps (1-100, default: 100)                  │
│  │   └── model (default: "o4-mini")                        │
│  │                                                         │
│  🔄 Processing Pipeline:                                    │
│  ├── 1. News Search (SERP API)                            │
│  ├── 2. Content Extraction (Browserless)                  │
│  ├── 3. AI Summarization (OpenAI)                         │
│  ├── 4. Content Aggregation                               │
│  └── 5. Response Delivery                                 │
│                                                            │
│  🔧 Tool Functions:                                        │
│  ├── search_news() - Google-style news queries           │
│  ├── fetch_content() - Article scraping & extraction     │
│  ├── perform_search() - SERP API integration             │
│  ├── scrape_content() - Content extraction & parsing     │
│  └── summarize_article() - AI-powered summarization      │
│                                                            │
│  📤 Output Formats:                                        │
│  ├── Synchronous: Direct JSON response                    │
│  ├── Asynchronous: Background processing + Bubble API     │
│  └── Debug Mode: Raw search results                       │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow Architecture

### 1. Request Processing

```
User Request → API Gateway → Process Handler
                              ├── Sync Processing (no user param)
                              └── Async Processing (with user param)
                                  ├── Send 202 Accepted
                                  ├── Background Task
                                  └── Status Updates
```

### 2. News Aggregation Pipeline

```
Topic Input
    ↓
🔍 Search Phase (max 10 calls)
    ├── search_news(topic, date_range)
    ├── SERP API Integration
    └── Results Filtering (5 per search)
    ↓
📄 Content Extraction Phase (3-7 calls)
    ├── fetch_content(url)
    ├── Browserless Scraping
    ├── Readability Processing
    └── Image Extraction
    ↓
🤖 AI Processing Phase
    ├── Article Summarization (GPT-4.1)
    ├── Content Aggregation
    └── Briefing Generation (o4-mini/others)
    ↓
📋 Output Generation
    ├── 10-20 Paragraph Briefing
    ├── Structured JSON Response
    └── Article Metadata
```

## 🔧 Technical Layout

### Core Functions

#### API Endpoint
- **Route**: `GET /news`
- **Handler**: `get_news()` → `process_news_request()`
- **Response Types**: 
  - `200 OK`: Synchronous processing complete
  - `202 Accepted`: Asynchronous processing started
  - `500 Error`: Processing failure

#### Search & Extraction Tools
```python
# News Search Tool
search_news(topic: str, date_range: str)
├── SERP API Integration
├── Query Optimization
└── Result Limiting (5 articles per search)

# Content Extraction Tool  
fetch_content(url: str)
├── Browserless Scraping
├── Readability Processing
├── Image Detection (OG tags + first img)
└── Content Summarization
```

#### AI Integration
```python
# Summarization Engine
summarize_article(title: str, text: str)
├── GPT-4.1 Processing
├── 3-sentence summaries
└── Information density optimization

# Main AI Agent
├── System Prompt: Investigative correspondent
├── Tools: search_news, fetch_content
├── Constraints: 48-hour window, 10 search limit
└── Output: Structured briefing
```

## 📱 Integration Points

### External APIs
- **SERP API**: Google-style news search
- **Browserless**: Web scraping and content extraction
- **OpenAI**: AI summarization and briefing generation
- **Bubble API**: Status updates and result delivery

### Status Management
```python
# Status Update Flow (Async Mode)
Initial Status → Processing Updates → Completion/Error
     ↓              ↓                    ↓
   "started"    "processing"       "completed"/"error"
   Progress: 0%  Progress: 1-99%   Progress: 100%
```

## 🔄 Response Structure

### Synchronous Response
```json
{
  "summary": "10-20 paragraph briefing with inline links",
  "articles": [
    {
      "url": "https://example.com/article",
      "title": "Article Title",
      "text": "Full article content",
      "image": "https://example.com/image.jpg"
    }
  ]
}
```

### Asynchronous Response (Initial)
```json
{
  "status": "accepted",
  "message": "Request accepted and being processed",
  "user": "user_id",
  "topic": "requested_topic"
}
```

## 🚀 Deployment Layout

### Infrastructure
- **Platform**: Fly.io
- **App Name**: `news-8-7pgg`
- **Region**: Dallas (dfw)
- **Scaling**: 2+ machines, auto-start/stop
- **Resources**: 1GB RAM, shared CPU

### Configuration Files
- `fly.toml` - Deployment configuration
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project metadata
- `Dockerfile` - Container setup

## 🔐 Environment Variables

```bash
OPENAI_API_KEY=<OpenAI API key>
SERP_API_KEY=<SerpAPI key>
BROWSERLESS_API_KEY=<Browserless token>
```

## 🎯 Key Features

### Content Processing
- **Smart Filtering**: Excludes stories from previous summaries
- **Multi-angle Coverage**: Strategic search planning
- **Rich Extraction**: Title, content, images, metadata
- **AI Summarization**: Concise, fact-focused briefings

### Performance Optimizations
- **Search Limits**: Maximum 10 news searches per request
- **Content Limits**: 3-7 article extractions per request
- **Time Windows**: Strict 48-hour news recency
- **Async Processing**: Background tasks for user requests

### Error Handling
- **Graceful Degradation**: Continues processing on partial failures
- **Status Reporting**: Real-time progress updates
- **Timeout Management**: Maximum step limits
- **API Integration**: Bubble API error reporting

This layout provides a robust, scalable news aggregation system that combines intelligent search, content extraction, and AI-powered summarization into coherent, actionable briefings.