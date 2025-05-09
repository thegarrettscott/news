import os
import json
import requests
import re
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

try:
    from readability import Document
except ModuleNotFoundError:
    raise ImportError("The 'readability' package requires a working Python SSL module. Run: apt install libssl-dev")

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")

def perform_search(topic: str, date_range: str):
    params = {
        "engine": "google",
        "q": topic,
        "api_key": SERP_API_KEY,
        "hl": "en",
        "gl": "us",
        "num": 10
    }
    if "day" in date_range or "days" in date_range:
        num = [int(s) for s in date_range.split() if s.isdigit()]
        days = num[0] if num else 1
        params["as_qdr"] = f"d{days}"

    res = requests.get("https://serpapi.com/search.json", params=params)
    data = res.json()
    results = []
    for item in data.get("organic_results", []):
        link = item.get("link")
        snippet = item.get("snippet") or item.get("title")
        date = item.get("date")
        if link and snippet:
            results.append({
                "link": link, 
                "preview": snippet,
                "date": date
            })
        if len(results) >= 5:
            break
    return {"results": results}

def scrape_content(url: str):
    bl_endpoint = f"https://chrome.browserless.io/content?token={BROWSERLESS_API_KEY}"
    try:
        res = requests.post(bl_endpoint, json={"url": url}, timeout=15)
        html = res.text
    except Exception as e:
        return {"url": url, "error": str(e), "title": None, "text": None, "image": None}

    doc = Document(html)
    title = doc.short_title()
    content_html = doc.summary()
    text_content = re.sub(r'<[^>]+>', '', content_html).strip()

    og_image = re.search(r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"', html, re.IGNORECASE)
    first_img = re.search(r'<img[^>]+src="([^"]+)"', content_html)
    image = og_image.group(1) if og_image else (first_img.group(1) if first_img else None)

    return {"url": url, "title": title, "text": text_content, "image": image}

tools = [
    {
        "type": "function",
        "name": "search_news",
        "description": "Search recent articles about a topic.",
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
    }
]

@app.get("/news", response_class=JSONResponse)
async def get_news(
    topic: str,
    date_range: str = "past 2 days",
    effort: str = Query(default="medium", enum=["low", "medium", "high"]),
    debug: bool = False,
    previous_summary: str = None,
    max_steps: int = Query(default=100, ge=1, le=100)
):
    # Cost calculation parameters
    INPUT_COST_PER_MILLION = 10.00  # $10.00 per 1M tokens for input
    CACHED_COST_PER_MILLION = 2.50  # $2.50 per 1M tokens for cached input
    OUTPUT_COST_PER_MILLION = 40.00  # $40.00 per 1M tokens for output
    
    input_messages = [
        {
            "role": "system",
            "content": (
                "You are a research correspondant that helps a newsletter writer gather data. "
                "Use tools to search and scrape the web. Return a long,10-20 paragraph detailed report "
                "with hyperlinks, image links, and summaries formatted for a newsletter writer. BE VERY CAREFUL TO ONLY INCLUDE NEWS FROM THE INCLUDED DATE RANGE."
            )
        }
    ]

    # Construct user message with optional previous summary
    user_message = f"Summarize recent news about {topic} from {date_range}."
    if previous_summary:
        user_message += f"\n\nHere is yesterday's newsletter summary for reference. Please ensure today's summary excludes these stories already covered:\n\n{previous_summary}"

    input_messages.append({"role": "user", "content": user_message})

    # If debug is True, return raw SERP results
    if debug:
        search_results = perform_search(topic, date_range)
        return search_results

    # Store scraped articles
    scraped_articles = []
    
    # Initialize token tracking variables at the start of the function
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0

    for step in range(max_steps):
        res = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "o3",
                "input": input_messages,
                "tools": tools,
                "reasoning": {"effort": effort},
                "tool_choice": "auto"
            }
        )
        
        print("Response Status Code:", res.status_code)
        print("Response Headers:", dict(res.headers))
        print("Raw Response Text:", res.text)
        
        data = res.json()
        print("Step", step, "Response Data:", json.dumps(data, indent=2))
        
        # Extract token usage from the API response
        if "usage" in data:
            print("Found usage in data root")
            if step == 0:
                total_input_tokens += data["usage"].get("prompt_tokens", 0)
            else:
                total_cached_tokens += data["usage"].get("prompt_tokens", 0)
            total_output_tokens += data["usage"].get("completion_tokens", 0)
        elif "metadata" in data and "usage" in data["metadata"]:
            print("Found usage in metadata")
            if step == 0:
                total_input_tokens += data["metadata"]["usage"].get("prompt_tokens", 0)
            else:
                total_cached_tokens += data["metadata"]["usage"].get("prompt_tokens", 0)
            total_output_tokens += data["metadata"]["usage"].get("completion_tokens", 0)
        elif "metadata" in data and "token_usage" in data["metadata"]:
            print("Found token_usage in metadata")
            if step == 0:
                total_input_tokens += data["metadata"]["token_usage"].get("prompt_tokens", 0)
            else:
                total_cached_tokens += data["metadata"]["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += data["metadata"]["token_usage"].get("completion_tokens", 0)
        
        print("Step", step, "Token Usage - Input:", total_input_tokens, "Output:", total_output_tokens, "Cached:", total_cached_tokens)
        
        # Calculate final costs
        input_cost = total_input_tokens * 0.01 / 1000  # $0.01 per 1K tokens
        output_cost = total_output_tokens * 0.03 / 1000  # $0.03 per 1K tokens
        cached_cost = total_cached_tokens * 0.01 / 1000  # $0.01 per 1K tokens
        total_cost = input_cost + output_cost + cached_cost
        
        cost_breakdown = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cached_input_tokens": total_cached_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "cached_cost": round(cached_cost, 6),
            "total_cost": round(total_cost, 6)
        }
        
        print("Final Cost Breakdown:", cost_breakdown)
        
        # Debug logging for entire response structure
        print(f"Step {step} full response data:", json.dumps(data, indent=2))
        
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
                result = (
                    perform_search(**args)
                    if item["name"] == "search_news"
                    else scrape_content(**args)
                )

                # Store scraped articles
                if item["name"] == "fetch_content":
                    scraped_articles.append(result)

                input_messages.append({
                    "type": "function_call",
                    "id": item["id"],
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"]
                })

                input_messages.append({
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": json.dumps(result)
                })

            elif item["type"] == "message":
                return {
                    "summary": item["content"],
                    "articles": scraped_articles,
                    "cost_breakdown": cost_breakdown
                }

    return JSONResponse(status_code=500, content={"error": f"Failed to generate summary after {max_steps} steps."})
