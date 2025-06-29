import os
import json
import requests
import re
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Dict, List, Optional

try:
    from readability import Document
except ModuleNotFoundError:
    raise ImportError("The 'readability' package requires a working Python SSL module. Run: apt install libssl-dev")

# Load environment variables
load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")

# Cost tracking configuration
COST_TRACKING = {
    "openai": {
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4.1": {"input": 0.01, "output": 0.03},  # per 1K tokens
        "o3": {"input": 0.005, "output": 0.015},  # per 1K tokens
        "o4-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
    },
    "xai": {
        "grok-3-latest": {"input": 0.0001, "output": 0.0003},  # per 1K tokens
        "grok-3-mini-latest": {"input": 0.00005, "output": 0.00015},  # per 1K tokens
        "o3": {"input": 0.005, "output": 0.015},  # per 1K tokens
        "o4-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
    },
    "browserless": {
        "per_request": 0.001  # per request
    }
}

class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.calls = []
        self.start_time = datetime.now()
    
    def add_call(self, function_name: str, model: str, provider: str, 
                 input_tokens: int = 0, output_tokens: int = 0, 
                 additional_cost: float = 0.0, success: bool = True):
        """Track a function call with cost calculation."""
        call_cost = 0.0
        
        # Calculate token costs
        if provider in COST_TRACKING and model in COST_TRACKING[provider]:
            pricing = COST_TRACKING[provider][model]
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]
            call_cost = input_cost + output_cost
        
        # Add additional costs (e.g., browserless)
        call_cost += additional_cost
        
        call_data = {
            "function": function_name,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": call_cost,
            "timestamp": datetime.now().isoformat(),
            "success": success
        }
        
        self.calls.append(call_data)
        self.total_cost += call_cost
        
        print(f"💰 Cost tracking: {function_name} ({model}) - ${call_cost:.4f} "
              f"({input_tokens} input, {output_tokens} output tokens)")
        
        return call_data
    
    def get_summary(self) -> Dict:
        """Get cost summary and breakdown."""
        provider_breakdown = {}
        function_breakdown = {}
        
        for call in self.calls:
            # Provider breakdown
            provider = call["provider"]
            if provider not in provider_breakdown:
                provider_breakdown[provider] = {"cost": 0.0, "calls": 0}
            provider_breakdown[provider]["cost"] += call["cost"]
            provider_breakdown[provider]["calls"] += 1
            
            # Function breakdown
            function = call["function"]
            if function not in function_breakdown:
                function_breakdown[function] = {"cost": 0.0, "calls": 0}
            function_breakdown[function]["cost"] += call["cost"]
            function_breakdown[function]["calls"] += 1
        
        return {
            "total_cost": self.total_cost,
            "total_calls": len(self.calls),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "provider_breakdown": provider_breakdown,
            "function_breakdown": function_breakdown,
            "calls": self.calls
        }

# Global cost tracker instance
cost_tracker = CostTracker()

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 characters per token is a common approximation)."""
    return len(text) // 4

def get_date_range(days: int):
    """Calculate date range for X.AI search."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def perform_search(topic: str, date_range: str, effort: str):
    """Search using X.AI's search API."""
    print(f"Starting perform_search for topic: {topic}, date_range: {date_range}, effort: {effort}")
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Calculate date range
    if "day" in date_range or "days" in date_range:
        num = [int(s) for s in date_range.split() if s.isdigit()]
        days = num[0] if num else 1
    else:
        days = 2  # default to 2 days
    
    from_date, to_date = get_date_range(days)
    print(f"Date range: {from_date} to {to_date}")
    
    # Determine the model based on effort
    if effort == "low":
        model = "o4-mini"
        search_model = "grok-3-mini-latest"
    elif effort == "high":
        model = "o3"
        search_model = "grok-3-latest"
    else:
        model = "o3"
        search_model = "grok-3-latest"

    # Always set the effort for /responses to 'high'
    response_effort = "high"

    # Update the payload for search_news to use the search_model - FOCUS ON X/TWITTER ONLY
    payload = {
        "model": search_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert social media curator specializing in X (formerly Twitter) content analysis. Your mission is to find the most compelling tweets, X posts, conversations, and social media discourse on the given topic. Focus on:\n\n- Viral tweets and trending conversations\n- Posts from verified accounts, influencers, experts, and officials\n- Tweet threads with detailed analysis or breaking news\n- Real-time reactions and public sentiment\n- Quote tweets and reply chains that show different perspectives\n- Trending hashtags and their context\n- X Spaces discussions and live commentary\n- Breaking news as it unfolds on X\n- Expert takes and professional commentary\n- Community reactions and grassroots conversations\n\nAlways include tweet text, author handles, engagement metrics when available, and direct links to the original posts."
            },
            {
                "role": "user",
                "content": f"Search X (Twitter) for the most important posts, tweets, and conversations about '{topic}' from the last {days} days. Find:\n\n1. Viral tweets and high-engagement posts\n2. Posts from verified accounts and experts\n3. Breaking news and real-time updates\n4. Tweet threads with detailed analysis\n5. Different perspectives and debates\n6. Trending hashtags related to the topic\n7. Quote tweets and meaningful replies\n8. Official statements posted on X\n9. Live reactions and commentary\n10. Community discussions and grassroots voices\n\nFocus EXCLUSIVELY on content from X/Twitter. Include tweet text, author information, and engagement data when possible."
            }
        ],
        "search_parameters": {
            "mode": "on",
            "sources": [
                {"type": "x"}
            ],
            "from_date": from_date,
            "to_date": to_date,
            "max_search_results": 30,
            "return_citations": True,
            "search_depth": "comprehensive"
        },
        "temperature": 0.3,
        "max_tokens": 3000
    }
    
    # Estimate input tokens for cost tracking
    input_text = payload["messages"][0]["content"] + payload["messages"][1]["content"]
    estimated_input_tokens = estimate_tokens(input_text)
    
    print("Making request to X.AI API...")
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        print(f"X.AI API response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        # Extract content and citations from X.AI response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        
        # Get actual token usage from response
        usage = data.get("usage", {})
        actual_input_tokens = usage.get("prompt_tokens", estimated_input_tokens)
        actual_output_tokens = usage.get("completion_tokens", estimate_tokens(content))
        
        # Track cost
        cost_tracker.add_call(
            function_name="search_news",
            model=search_model,
            provider="xai",
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            success=True
        )
        
        print(f"Found {len(citations)} citations")
        
        return {
            "content": content,
            "citations": citations
        }
        
    except Exception as e:
        print(f"Error in X.AI search: {str(e)}")
        
        # Track failed call
        cost_tracker.add_call(
            function_name="search_news",
            model=search_model,
            provider="xai",
            input_tokens=estimated_input_tokens,
            output_tokens=0,
            success=False
        )
        
        return {"content": "", "citations": [], "error": str(e)}

def dig_deeper(story: str, days: int = 2, additional_focus: str = ""):
    """Perform a deep dive into a specific story using X.AI."""
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    from_date, to_date = get_date_range(days)
    
    payload = {
        "model": "grok-3-mini-latest",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert investigative researcher specializing in deep-dive story analysis. Your mission is to thoroughly dissect a specific news story by gathering comprehensive information, multiple perspectives, expert opinions, and contextual background. You should:\n\n- Find and extract direct quotes from key players, officials, experts, and affected parties\n- Identify different viewpoints and conflicting opinions on the story\n- Uncover background context and related events that led to this story\n- Surface expert analysis and commentary from credible sources, especcialy from X \n- Look for follow-up developments, reactions, and consequences\n- Find data, statistics, and factual details that support or contradict claims\n- Identify stakeholders and their positions on the issue\n- Gather social media reactions and public sentiment\n- Find related stories or similar cases for comparison\n- Present information in a structured way with clear attribution and sourcing and include links to images that would be could pictures to use in the newsletter."
            },
            {
                "role": "user",
                "content": f"Deep dive into this specific story: \"{story}\"\n\nI need you to research every angle of this story. Find:\n1. All key quotes from officials, experts, and involved parties\n2. Different perspectives and opinions (both supporting and opposing)\n3. Background context and timeline of events\n4. Data, statistics, and factual claims with verification\n5. Expert analysis and commentary\n6. Public and social media reactions\n7. Follow-up developments and consequences\n8. Related stories or precedents\n9. Stakeholder positions and motivations\n10. Any controversies or disputed facts\n\nFocus on stories from the last {days} days but include relevant background context from earlier if needed. {additional_focus}"
            }
        ],
        "search_parameters": {
            "mode": "on",
            "sources": [
                {"type": "web"},
                {"type": "news"},
                {"type": "x"},
                {"type": "academic"}
            ],
            "from_date": from_date,
            "to_date": to_date,
            "max_search_results": 40,
            "return_citations": True,
            "search_depth": "comprehensive"
        },
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract content and citations from X.AI response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        
        return {
            "content": content,
            "citations": citations
        }
    except Exception as e:
        print(f"Error in X.AI deep dive: {str(e)}")
        return {"content": "", "citations": [], "error": str(e)}

def summarize_article(title: str, text: str) -> str:
    """Summarize an article into 3 concise, information-dense sentences using GPT-4.1."""
    prompt = f"""Summarize the following article into exactly 3 concise, information-dense sentences. Focus on key facts, figures, and implications:

Title: {title}
Content: {text}

Summary:"""
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 150
        }
    )
    
    if response.status_code != 200:
        return f"Error summarizing article: {response.text}"
    
    return response.json()["choices"][0]["message"]["content"].strip()

def scrape_content(url: str):
    bl_endpoint = f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}"
    try:
        res = requests.post(bl_endpoint, json={"url": url}, timeout=15)
        if res.status_code == 429:  # Rate limit exceeded
            print(f"Browserless.io rate limit exceeded for URL: {url}")
            
            # Track rate limit as a cost (still counts as a request)
            cost_tracker.add_call(
                function_name="fetch_content",
                model="browserless",
                provider="browserless",
                additional_cost=COST_TRACKING["browserless"]["per_request"],
                success=False
            )
            
            return {
                "url": url,
                "title": None,
                "text": "Browserless.io rate limit exceeded. Please try again later.",
                "image": None,
                "metadata": {
                    "error": "rate_limit_exceeded",
                    "content_length": 0
                }
            }
        html = res.text
        
        # Track successful browserless call
        cost_tracker.add_call(
            function_name="fetch_content",
            model="browserless",
            provider="browserless",
            additional_cost=COST_TRACKING["browserless"]["per_request"],
            success=True
        )
        
    except Exception as e:
        print(f"Error scraping content for {url}: {str(e)}")
        
        # Track failed browserless call
        cost_tracker.add_call(
            function_name="fetch_content",
            model="browserless",
            provider="browserless",
            additional_cost=COST_TRACKING["browserless"]["per_request"],
            success=False
        )
        
        return {
            "url": url,
            "title": None,
            "text": f"Error scraping content: {str(e)}",
            "image": None,
            "metadata": {
                "error": str(e),
                "content_length": 0
            }
        }

    doc = Document(html)
    title = doc.short_title()
    content_html = doc.summary()
    text_content = re.sub(r'<[^>]+>', '', content_html).strip()

    og_image = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', html, re.IGNORECASE)
    first_img = re.search(r'<img[^>]+src="([^"]+)"', content_html)
    image = og_image.group(1) if og_image else (first_img.group(1) if first_img else None)

    # Create the full content object
    full_content = {
        "url": url,
        "title": title,
        "text": text_content,
        "image": image,
        "metadata": {
            "og_image": og_image.group(1) if og_image else None,
            "first_image": first_img.group(1) if first_img else None,
            "content_length": len(text_content) if text_content else 0
        }
    }

    # Create a summarized version for o3
    if title and text_content:
        summary = summarize_article(title, text_content)
        return {
            "url": url,
            "title": title,
            "text": summary,
            "image": image,
            "_full_content": full_content  # Store full content for final response
        }
    
    return full_content

def openai_web_search(query: str, max_results: int = 10, focus_area: str = ""):
    """Perform web search using OpenAI's native search capability."""
    print(f"Starting OpenAI web search for query: {query}, max_results: {max_results}")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Enhanced system prompt for research-focused search
    system_prompt = """You are an expert research assistant specializing in comprehensive web search and analysis. Your role is to:

1. Conduct thorough web searches to find the most relevant, recent, and authoritative sources
2. Analyze and synthesize information from multiple sources
3. Identify key facts, quotes, and data points
4. Provide clear attribution and source links
5. Flag any conflicting information or different perspectives
6. Focus on credible news sources, official statements, and expert analysis
7. Extract actionable insights and important context

Always prioritize accuracy, recency, and source credibility in your research."""

    user_prompt = f"""Research: {query}

Please conduct a comprehensive web search and provide:
1. Key findings and developments
2. Important quotes from officials, experts, or key figures
3. Relevant data, statistics, and facts
4. Multiple perspectives if they exist
5. Source links for verification
6. Timeline of events if applicable

{f"Special focus: {focus_area}" if focus_area else ""}

Prioritize sources from the last 48 hours but include essential background context as needed."""

    payload = {
        "model": "gpt-4o",  # Use the latest model with search capabilities
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "max_results": max_results
                }
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens": 3000
    }
    
    # Estimate input tokens for cost tracking
    input_text = system_prompt + user_prompt
    estimated_input_tokens = estimate_tokens(input_text)
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        print(f"OpenAI web search response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        # Extract the response content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract any tool calls and their results
        tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        search_results = []
        
        for tool_call in tool_calls:
            if tool_call.get("type") == "web_search":
                search_results.append(tool_call)
        
        # Get actual token usage from response
        usage = data.get("usage", {})
        actual_input_tokens = usage.get("prompt_tokens", estimated_input_tokens)
        actual_output_tokens = usage.get("completion_tokens", estimate_tokens(content))
        
        # Track cost
        cost_tracker.add_call(
            function_name="openai_web_search",
            model="gpt-4o",
            provider="openai",
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            success=True
        )
        
        return {
            "content": content,
            "search_results": search_results,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in OpenAI web search: {str(e)}")
        
        # Track failed call
        cost_tracker.add_call(
            function_name="openai_web_search",
            model="gpt-4o",
            provider="openai",
            input_tokens=estimated_input_tokens,
            output_tokens=0,
            success=False
        )
        
        return {"content": "", "search_results": [], "error": str(e), "query": query}

def deep_research_analysis(topic: str, research_depth: str = "comprehensive", time_range: str = "48 hours"):
    """Conduct deep research analysis using OpenAI's capabilities."""
    print(f"Starting deep research analysis for topic: {topic}")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are an expert investigative researcher conducting deep analysis on complex topics. Your approach should be:

1. SYSTEMATIC: Break down the topic into key components and research each thoroughly
2. MULTI-PERSPECTIVE: Seek out different viewpoints, expert opinions, and stakeholder positions  
3. FACT-FOCUSED: Prioritize verifiable facts, official statements, and credible sources
4. CONTEXTUAL: Provide historical background and broader implications
5. ANALYTICAL: Connect dots between related events and identify patterns
6. COMPREHENSIVE: Leave no stone unturned in your investigation

Use web search extensively to gather information from multiple authoritative sources. Always cite your sources and flag any conflicting information you find."""

    user_prompt = f"""Conduct a comprehensive deep research analysis on: {topic}

Research Requirements:
- Time focus: {time_range}
- Depth level: {research_depth}

Please investigate and provide:

1. **Current Status & Latest Developments**
   - What's happening right now?
   - Latest news and updates
   - Recent official statements

2. **Key Players & Stakeholders**
   - Who are the main actors?
   - What are their positions and motivations?
   - Direct quotes from key figures

3. **Background & Context**
   - How did we get here?
   - Historical precedents
   - Related events and connections

4. **Data & Evidence**
   - Hard facts and statistics
   - Financial figures where relevant
   - Technical details and specifications

5. **Multiple Perspectives**
   - Different viewpoints on the issue
   - Expert analysis and commentary
   - Public and industry reactions

6. **Implications & Future Outlook**
   - What does this mean going forward?
   - Potential consequences and outcomes
   - Timeline of expected developments

Use extensive web searches to gather comprehensive information. Prioritize recent sources but include essential background context."""

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "max_results": 20
                }
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "topic": topic,
            "research_depth": research_depth,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in deep research analysis: {str(e)}")
        return {"content": "", "tool_calls": [], "error": str(e), "topic": topic}

tools = [
    {
        "type": "function",
        "name": "search_news",
        "description": "Search X (Twitter) for viral tweets, trending conversations, and social media discourse on a specific topic. Returns tweet content, author information, engagement metrics, and direct links to posts.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": { "type": "string" },
                "date_range": { "type": "string" }
            },
            "required": ["topic", "date_range"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "dig_deeper",
        "description": "Perform a deep dive analysis of a specific story.",
        "parameters": {
            "type": "object",
            "properties": {
                "story": { "type": "string" },
                "days": { "type": "integer", "default": 2 },
                "additional_focus": { "type": "string", "default": "" }
            },
            "required": ["story"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "fetch_content",
        "description": "Scrape full content of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": { "type": "string" }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "openai_web_search",
        "description": "Perform comprehensive web search using OpenAI's native search capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "max_results": { "type": "integer", "default": 10 },
                "focus_area": { "type": "string", "default": "" }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    {
        "type": "function", 
        "name": "deep_research_analysis",
        "description": "Conduct comprehensive deep research analysis on complex topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": { "type": "string" },
                "research_depth": { "type": "string", "enum": ["basic", "comprehensive", "exhaustive"], "default": "comprehensive" },
                "time_range": { "type": "string", "default": "48 hours" }
            },
            "required": ["topic"],
            "additionalProperties": False
        }
    }
]

@app.get("/news", response_class=JSONResponse)
async def get_news(
    background_tasks: BackgroundTasks,
    topic: str,
    user: str = None,
    date_range: str = "past 2 days",
    effort: str = Query(default="medium", enum=["low", "medium", "high"]),
    debug: bool = False,
    previous_summary: str = None,
    max_steps: int = Query(default=100, ge=1, le=100)
):
    print(f"Received request - Topic: {topic}, User: {user}, Effort: {effort}, Debug: {debug}")
    
    # If user is provided, send acceptance response but continue processing
    if user:
        print(f"User provided: {user}, sending initial status update")
        # Send initial status update
        try:
            status_response = requests.post(
                "https://yousletter.bubbleapps.io/api/1.1/wf/status_update_api",
                json={
                    "user": user,
                    "status": "started",
                    "message": "Starting news aggregation process",
                    "progress": 0
                },
                timeout=5
            )
            print(f"Status update response: {status_response.status_code} - {status_response.text}")
        except Exception as e:
            print(f"Failed to send initial status update: {e}")

        print("Sending 202 Accepted response and continuing processing in background")
        # Add the processing to background tasks
        background_tasks.add_task(process_news_request, topic, user, date_range, effort, debug, previous_summary, max_steps)
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "Your request has been accepted and is being processed. The results will be sent to the Bubble API.",
                "user": user,
                "topic": topic
            }
        )

    # If no user provided, process synchronously
    return await process_news_request(topic, user, date_range, effort, debug, previous_summary, max_steps)

async def process_news_request(topic: str, user: str, date_range: str, effort: str, debug: bool, previous_summary: str, max_steps: int):
    input_messages = [
        {
            "role": "system",
            "content": (
                """YOU ARE AN INVESTIGATIVE RESEARCH CORRESPONDENT WORKING FOR A HUMAN NEWSLETTER WRITER.
YOUR SOLE JOB: surface the most important, factual news published in the last 48 hours on a given topic and deliver a ready-to-use briefing.

────────────────────────────────────────
I. MINDSET & APPROACH
────────────────────────────────────────

Treat yourself as a rigorously trained journalist. Be skeptical, concise, and proof-driven.

Think first, act second. Before every tool call, pause to decide exactly what you need and why.

Optimize for signal-to-noise. A shorter, cleaner hit list beats a bloated dump of links.

Your reader is time-poor. They want facts, figures, original quotes, and links—no opinion, no spin.

Always cross-check high-impact claims with two independent outlets.

────────────────────────────────────────
II. INPUTS YOU RECEIVE EACH RUN
────────────────────────────────────────
• topic – the subject to cover
• previous_summary – a text blob listing what was covered yesterday (may be empty)

────────────────────────────────────────
III. TOOLS AVAILABLE – AND NOTHING ELSE
────────────────────────────────────────
search_news(topic, date_range)
– X.AI-powered search focused EXCLUSIVELY on X (Twitter) content.
– Returns viral tweets, trending conversations, expert posts, and real-time social discourse.
– Includes tweet text, author handles, engagement metrics, and direct post links.
– date_range must be "past 2 days" or narrower.
– MAX TEN calls per assignment.

dig_deeper(story, days, additional_focus)
– Follow-up research on a specific story string using X.AI.
– days defines how far back to examine (keep ≤ 2).
– additional_focus lets you narrow: e.g., "financials", "lawsuit source docs".

fetch_content(url)
– Scrapes full article: title, plain text body, lead image (url, caption, alt).
– REQUIRED on the five to seven most critical articles.

openai_web_search(query, max_results, focus_area)
– OpenAI's native web search with real-time access to current information.
– Provides comprehensive analysis with source attribution.
– Use for broad topic research and fact verification.
– max_results: 10-20 for comprehensive coverage.

deep_research_analysis(topic, research_depth, time_range)
– Comprehensive investigative research using OpenAI's advanced capabilities.
– research_depth: "basic", "comprehensive", or "exhaustive".
– Provides structured analysis with multiple perspectives.
– Use for complex topics requiring deep investigation.

NEVER mention any tool names in the briefing itself. The writer only sees final copy.

────────────────────────────────────────
IV. HOW TO PROMPT THE TOOLS EFFECTIVELY
────────────────────────────────────────
A. search_news best practices
• Be curious about the topic and start broadly, starting always with just searching the topic verbatim and then narrow down to specific angles. Remember try to be a really good investigative reporter and try to find the most interesting stories, while also making sure not to miss big stories.

B. dig_deeper best practices
• Trigger after the broad query has been run and you have a list of interesting stories.
• Clarify what you still need: source documents? rival viewpoint? online opiniona? timeline? Ask for as much as possible, you won't keep it all, but you should still ask.
• Keep days ≤ 2 so all follow-ups remain inside the 48-hour window.
• Example call: dig_deeper("NVIDIA's AI chip shortage", 2, "online opinions, CEO Tweets, contrarian viewpoints, NVIDEO statement").

C. fetch_content best practices
• Choose definitive, original-reporting sources first (major newspapers, wires, specialist trades).
• Always call on investigative exclusives, breaking regulatory filings, and any piece with crucial numbers.
• Example call: fetch_content("https://www.ft.com/content/abcdef").

────────────────────────────────────────
V. END-TO-END RESEARCH WORKFLOW
────────────────────────────────────────
STEP 0 QUICK SCAN OF previous_summary
• List yesterday's slugs to avoid duplicates.
• Flag any story that might have materially changed today.

STEP 1 PLAN YOUR RESEARCH STRATEGY
- Start with strategic tool selection:
• For broad topic coverage: Use openai_web_search for comprehensive, real-time information
• For X/Twitter social discourse: Use search_news for viral tweets, trending conversations, and real-time reactions
• For complex investigations: Use deep_research_analysis for thorough investigation
• Break the topic into 3-5 sub-queries covering different angles
• Draft ≤ 10 total search calls across all tools
• Write them down before executing; this prevents wasteful calls.

STEP 2 EXECUTE RESEARCH CALLS
• Start with 1-2 openai_web_search calls for broad topic understanding
• Follow with targeted search_news calls for X/Twitter social media coverage and viral conversations
• Use deep_research_analysis for complex stories requiring investigation
• Collect tweets, posts, headlines, sources, timestamps, and analysis
• Immediately discard anything older than 48 h or clearly duplicative
• For each result, jot a one-line note on why it might matter.

STEP 3 TRIAGE HITS
• Select top ~10 hits for deeper validation.

STEP 4 dig_deeper WHERE NEEDED
• For any top-score item lacking detail, call dig_deeper to pull missing context.
• Merge new facts back into your notes.

STEP 5 fetch_content FOR CORE ARTICLES
• Pick 5-7 must-cover URLs t gather more info on important developments.
• Run fetch_content on each.
• While reading scraped text, extract:
– direct quotes with attribution
– numbers (funding amounts, fines, units sold)
– exact dates and future deadlines
– named entities (people, orgs, places)
– any dissenting or corroborating sources cited inside the piece
• Capture lead image metadata (file name or url, alt-text, caption) for inclusion later.

STEP 6 FACT CROSS-CHECK
• Verify high-impact numbers against at least one independent outlet or official doc.
• If conflict found, note both versions and which seems more credible.

STEP 7 DEDUP & UPDATE FILTER
• Remove stories fully covered in previous_summary unless there is a new development timestamped within the last 24 h.
• For updated threads, focus paragraphs on "what changed since yesterday."

STEP 8 STRUCTURE THE BRIEFING
• Aim for 40-50 paragraphs.
• Each paragraph starts with an ALL-CAPS slug of 4-10 characters, colon, space (e.g., M&A:, GOVT:, DATA:, LEGAL:, EARN:).
• First sentence: article title linked inline to its url. Immediately follow with outlet in parentheses.
• Body: 1-3 tightly written sentences summarizing the new facts, quoting numbers, citing named sources.
• End of paragraph: embed image if scraped: [image: filename alt:"..." caption:"..."].
• Keep paragraphs logically grouped: separate slug for each distinct development.

STEP 9 CLOSE WITH WHY IT MATTERS
• One sentence, plain language, summarizing why the day's developments are important for the newsletter audience.
• Do not introduce new info here.

STEP 10 FINAL SELF-CHECK
• Count search_news calls (≤ 10).
• Ensure at least 5 fetch_content calls.
• Confirm no headline older than 48 h.
• Verify no paragraph duplicates previous_summary.
• Proofread figures and quotes verbatim.
• Confirm each paragraph contains link, slug, and if available image metadata.

────────────────────────────────────────
VI. STYLE RULES FOR WRITING
────────────────────────────────────────
• Neutral, third-person, factual.
• No emojis, hype words, or editorial adjectives.
• Use present perfect or past tense for events; future tense only for scheduled events.
• Numbers: always include units (USD, %, units, miles, etc.).
• Quotes: short, direct, attributed ("We plan to expand," CEO Jane Doe told Reuters).
• Links: embed only on article titles; nowhere else.
• One paragraph = one idea. Keep sentences short.

────────────────────────────────────────
VII. HARD CONSTRAINTS
────────────────────────────────────────
✓ Strict 48-hour freshness window for every headline and data point.
✓ Maximum ten search_news calls.
✓ Minimum five and maximum 15 fetch_content calls.
✓ Exclude or succinctly update anything that appears in previous_summary.
✓ Stop research once briefing meets quality bar; do not exceed time or tool limits.
✓ Output must be the JSON object expected by downstream endpoint (fields: briefing_text).

────────────────────────────────────────
VIII. FAILURE MODES TO AVOID
────────────────────────────────────────
× Running search_news excessively with broad queries ("climate change").
× Including headlines > 48 h old.
× Rehashing yesterday's stories without a material new twist.
× Swamping the reader with 30+ paragraphs or burying key facts under fluff.
× Citing unverified social media rumors or single-source claims without confirmation.
× Forgetting to provide lead image metadata when available.

────────────────────────────────────────
IX. BEST-PRACTICE EXAMPLE (ABBREVIATED)
────────────────────────────────────────
Suppose topic = "electric vehicle batteries" and previous_summary covered Ford-CATL licensing deal.

Plan queries:
Start with a broad query:
"electric vehicle batteries"
Then narrow down to specific angles based on the result
"solid-state battery pilot plant funding past 24 hours"

"lithium prices contract negotiations 'past 24 hours'"

"EV battery recycling startup Series B 'past 24 hours'"
Execute queries, triage results, dig_deeper on "DOE grants" for award amounts, fetch_content from a variety of sources.
Draft paragraphs:

TECH: [article title linked] (Nikkei Asia) Toyota said Wednesday it will begin mass-production of solid-state EV batteries in 2027, aiming for 1,000-km range and 10-minute recharge, executives told reporters after unveiling a pilot line in Aichi. The ¥1.5 tn ($9.5 bn) project is partly funded by Japan's Green Innovation fund. [image: toyota_solidstate.jpg alt:"Prototype solid-state cell" caption:"Toyota's pilot line cell"]

FIN: ...

Close with: Why it matters: Cheaper, denser batteries arriving by 2027 could slash EV sticker prices and reshape supply chains across Asia, Europe, and the US.

────────────────────────────────────────
X. FINAL DELIVERY FORMAT (NO MARKDOWN)
Return a JSON object with one key:
{ "briefing_text": "<full briefing exactly as drafted above>" }
Do not wrap the JSON in code fences.

Follow these instructions meticulously and you will consistently produce high-quality, ready-to-publish briefings that save the newsletter writer hours of work."""
            )
        }
    ]

    # Construct user message with optional previous summary
    user_message = f"Summarize recent news about {topic} from {date_range}."
    if previous_summary:
        user_message += f"\n\nHere is yesterday's newsletter summary for reference. Please ensure today's summary excludes these stories already covered:\n\n{previous_summary}"

    input_messages.append({"role": "user", "content": user_message})

    # Store scraped articles
    scraped_articles = []
    full_articles = []  # Store full article content

    # Determine the model internally based on effort (low -> o4-mini, medium/high -> o3)
    selected_model = "o4-mini" if effort == "low" else "o3"

    for step in range(max_steps):
        # Update status for each major step
        if user:
            try:
                requests.post(
                    "https://yousletter.bubbleapps.io/api/1.1/wf/status_update_api",
                    json={
                        "user": user,
                        "status": "processing",
                        "message": f"Processing step {step + 1} of {max_steps}",
                        "progress": int((step + 1) / max_steps * 100)
                    }
                )
            except Exception as e:
                print(f"Failed to send status update: {e}")

        res = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": selected_model,
                "input": input_messages,
                "tools": tools,
                "reasoning": {"effort": "high"},
                "tool_choice": "auto"
            }
        )
        
        if res.status_code != 200:
            return JSONResponse(status_code=500, content={
                "error": {
                    "code": res.status_code,
                    "message": res.text
                }
            })

        data = res.json()
        outputs = data.get("output", [])
        for item in outputs:
            if item["type"] == "reasoning":
                input_messages.append({
                    "type": "reasoning",
                    "id": item["id"],
                    "summary": item.get("summary", [])
                })

            elif item["type"] == "function_call":
                args = json.loads(item["arguments"])
                if item["name"] == "search_news":
                    result = perform_search(**args, effort=effort)
                elif item["name"] == "dig_deeper":
                    result = dig_deeper(**args)
                elif item["name"] == "fetch_content":
                    result = scrape_content(**args)
                elif item["name"] == "openai_web_search":
                    result = openai_web_search(**args)
                elif item["name"] == "deep_research_analysis":
                    result = deep_research_analysis(**args)
                else:
                    result = {"error": f"Unknown function: {item['name']}"}

                # Store scraped articles
                if item["name"] == "fetch_content":
                    # Store the summarized version for o3
                    scraped_articles.append({
                        "url": result["url"],
                        "title": result["title"],
                        "text": result["text"],
                        "image": result["image"]
                    })
                    # Store the full content for the final response
                    if "_full_content" in result:
                        full_articles.append(result["_full_content"])
                    else:
                        # If no _full_content, use the basic content
                        full_articles.append({
                            "url": result["url"],
                            "title": result["title"],
                            "text": result["text"],
                            "image": result["image"],
                            "metadata": {
                                "content_length": len(result["text"]) if result["text"] else 0
                            }
                        })
                elif item["name"] == "dig_deeper":
                    # Store the deep dive results
                    if "choices" in result:
                        deep_dive_content = result["choices"][0]["message"]["content"]
                        citations = result.get("citations", [])
                        scraped_articles.append({
                            "type": "deep_dive",
                            "content": deep_dive_content,
                            "citations": citations
                        })

                input_messages.append({
                    "type": "function_call",
                    "id": item["id"],
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"]
                })

                # Format the result based on the function type
                if item["name"] in ["search_news", "dig_deeper"]:
                    formatted_result = {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps({
                            "content": result.get("content", ""),
                            "citations": result.get("citations", []),
                            "error": result.get("error")
                        })
                    }
                else:
                    formatted_result = {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(result)
                    }

                input_messages.append(formatted_result)

            elif item["type"] == "message":
                # Prepare the response data
                print(f"Debug: Number of full articles: {len(full_articles)}")
                print(f"Debug: Full articles content: {json.dumps(full_articles, indent=2)}")
                
                # If no articles were scraped, force a fetch_content call for the first citation in the last search_news result
                if not full_articles:
                    print("No articles found, attempting to fetch content for the first citation in the last search_news result.")
                    # Find the last search_news result in input_messages
                    for msg in reversed(input_messages):
                        if msg.get("type") == "function_call_output" and 'search_news' in msg.get("output", ""):
                            try:
                                output_data = json.loads(msg["output"])
                                citations = output_data.get("citations", [])
                                if citations and isinstance(citations[0], dict) and citations[0].get("url"):
                                    url = citations[0]["url"]
                                    print(f"Forcing fetch_content for URL: {url}")
                                    result = scrape_content(url)
                                    if "_full_content" in result:
                                        full_articles.append(result["_full_content"])
                                    else:
                                        full_articles.append({
                                            "url": result.get("url"),
                                            "title": result.get("title"),
                                            "text": result.get("text"),
                                            "image": result.get("image"),
                                            "metadata": {
                                                "content_length": len(result.get("text") or "")
                                            }
                                        })
                                break
                            except Exception as e:
                                print(f"Error forcing fetch_content: {e}")
                                break
                response_data = {
                    "summary": item["content"],
                    "articles": full_articles,  # Include the full article content
                    "cost_analysis": cost_tracker.get_summary(),
                    "debug": {
                        "scraped_articles": scraped_articles if debug else None,
                        "full_articles": full_articles if debug else None
                    } if debug else None
                }
                
                # Only send to Bubble API if user is provided
                if user:
                    # Send final status update
                    try:
                        requests.post(
                            "https://yousletter.bubbleapps.io/api/1.1/wf/status_update_api",
                            json={
                                "user": user,
                                "status": "completed",
                                "message": "News aggregation completed successfully",
                                "progress": 100
                            }
                        )
                    except Exception as e:
                        print(f"Failed to send final status update: {e}")

                    # Extract and clean the newsletter content
                    newsletter_content = item["content"]
                    
                    # Remove markdown code blocks and other formatting that might interfere
                    newsletter_content = re.sub(r'```html\s*', '', newsletter_content)
                    newsletter_content = re.sub(r'```\s*', '', newsletter_content)
                    newsletter_content = re.sub(r'`([^`]*)`', r'\1', newsletter_content)  # Remove inline code
                    newsletter_content = re.sub(r'\*\*([^*]*)\*\*', r'\1', newsletter_content)  # Remove bold
                    newsletter_content = re.sub(r'\*([^*]*)\*', r'\1', newsletter_content)  # Remove italic
                    newsletter_content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', newsletter_content)  # Remove markdown links, keep text
                    newsletter_content = re.sub(r'^\s*[-*+]\s+', '', newsletter_content, flags=re.MULTILINE)  # Remove list markers
                    newsletter_content = re.sub(r'^\s*\d+\.\s+', '', newsletter_content, flags=re.MULTILINE)  # Remove numbered list markers
                    newsletter_content = re.sub(r'^\s*#+\s+', '', newsletter_content, flags=re.MULTILINE)  # Remove headers
                    newsletter_content = re.sub(r'^\s*>\s+', '', newsletter_content, flags=re.MULTILINE)  # Remove blockquotes
                    newsletter_content = re.sub(r'^\s*\|.*\|.*$', '', newsletter_content, flags=re.MULTILINE)  # Remove table rows
                    newsletter_content = re.sub(r'^\s*[-=]+\s*$', '', newsletter_content, flags=re.MULTILINE)  # Remove horizontal rules
                    newsletter_content = re.sub(r'\n\s*\n\s*\n', '\n\n', newsletter_content)  # Normalize multiple newlines
                    newsletter_content = newsletter_content.strip()  # Remove leading/trailing whitespace
                    
                    # Convert newsletter content to base64
                    encoded_response = base64.b64encode(newsletter_content.encode()).decode()
                    
                    # Send to Bubble API
                    bubble_response = requests.post(
                        "https://yousletter.bubbleapps.io/api/1.1/wf/newsletter",
                        json={
                            "user": user,
                            "text": encoded_response
                        }
                    )
                    
                    if bubble_response.status_code != 200:
                        # Send error status update
                        try:
                            requests.post(
                                "https://yousletter.bubbleapps.io/api/1.1/wf/status_update_api",
                                json={
                                    "user": user,
                                    "status": "error",
                                    "message": f"Failed to send to Bubble API: {bubble_response.text}",
                                    "progress": 100
                                }
                            )
                        except Exception as e:
                            print(f"Failed to send error status update: {e}")

                        return JSONResponse(
                            status_code=500,
                            content={"error": f"Failed to send to Bubble API: {bubble_response.text}"}
                        )
                
                return response_data

    # Send timeout status update if we reach max steps
    if user:
        try:
            requests.post(
                "https://yousletter.bubbleapps.io/api/1.1/wf/status_update_api",
                json={
                    "user": user,
                    "status": "error",
                    "message": f"Failed to generate summary after {max_steps} steps",
                    "progress": 100
                }
            )
        except Exception as e:
            print(f"Failed to send timeout status update: {e}")

    return JSONResponse(status_code=500, content={"error": f"Failed to generate summary after {max_steps} steps."})

@app.get("/costs", response_class=JSONResponse)
async def get_cost_analysis():
    """Get cost analysis for the current session."""
    return cost_tracker.get_summary()

@app.get("/costs/reset", response_class=JSONResponse)
async def reset_cost_tracker():
    """Reset the cost tracker for a new session."""
    global cost_tracker
    cost_tracker = CostTracker()
    return {"message": "Cost tracker reset successfully"}