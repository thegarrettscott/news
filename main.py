import os
import json
import requests
import re
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

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
    previous_summary: str = None
):
    input_messages = [
        {
            "role": "system",
            "content": (
                "You are a research correspondant that helps a newsletter writer gather data. "
                "Use tools to search and scrape the web. Return a long, detailed report "
                "with hyperlinks and summaries formatted for a newsletter writer."
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

    async def generate_steps():
        all_steps = []
        for step in range(30):
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

            if res.status_code != 200:
                yield json.dumps({
                    "error": {
                        "code": res.status_code,
                        "message": res.text
                    }
                }) + "\n"
                return

            data = res.json()
            outputs = data.get("output", [])
            for item in outputs:
                if item["type"] == "reasoning":
                    input_messages.append({
                        "type": "reasoning",
                        "id": item["id"],
                        "summary": item.get("summary", [])
                    })
                    if debug:
                        step_data = {
                            "type": "reasoning",
                            "content": item.get("summary", [])
                        }
                        all_steps.append(step_data)
                        yield json.dumps(step_data) + "\n"

                elif item["type"] == "function_call":
                    args = json.loads(item["arguments"])
                    result = (
                        perform_search(**args)
                        if item["name"] == "search_news"
                        else scrape_content(**args)
                    )

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

                    if debug:
                        step_data = {
                            "type": "function_call",
                            "name": item["name"],
                            "arguments": args,
                            "result": result
                        }
                        all_steps.append(step_data)
                        yield json.dumps(step_data) + "\n"

                elif item["type"] == "message":
                    if debug:
                        step_data = {
                            "type": "final_message",
                            "content": item["content"]
                        }
                        all_steps.append(step_data)
                        yield json.dumps(step_data) + "\n"
                        return
                    return {"summary": item["content"]}

            await asyncio.sleep(0.1)  # Small delay between steps

        yield json.dumps({"error": "Failed to generate summary after 30 steps."}) + "\n"

    if debug:
        return StreamingResponse(
            generate_steps(),
            media_type="application/x-ndjson"
        )
    
    # For non-debug mode, collect all steps and return final result
    all_steps = []
    async for step in generate_steps():
        all_steps.append(json.loads(step))
        if step.get("type") == "final_message":
            return {"summary": step["content"]}
    
    return JSONResponse(status_code=500, content={"error": "Failed to generate summary after 30 steps."})
