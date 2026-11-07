from datetime import datetime
import json
import requests

from Utilities.utilities import log
from rag import search

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city: str):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text if response.status_code == 200 else "Could not retrieve weather information."

def get_city():
    return "Bangalore"

def search_knowledge_base(query):
    results = search(query, top_k=3)

    log(f"🔍 Search results for query: '{query}'")
    for r in results:
        log(f"  ⤷ Similarity: {r['similarity']:.4f}, Source: {r['source']}, Size: {r['size']}")
        log(f"      Text: {r['text'][:100]}...")  # Log first 100 characters of the text

    blocks = []
    for r in results:
        sourceFilename = r['source']
        chunkId = r['chunk_id'] if 'chunk_id' in r else None
        chunktext = r['text']

        header = f"[{sourceFilename} #{chunkId}]"
        block = f"{header}\n{chunktext}"
        blocks.append(block)

    return "\n\n".join(blocks)




# TOOLS = [{}, {}, ..]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time in the format YYYY-MM-DD HH:MM:SS.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "get_city_weather",
            "description": "Returns the current weather information for a specified city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city for which to retrieve weather information."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "get_city",
            "description": "Returns the name of the city.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function":{
            "name": "search_knowledge_base",
            "description": "Searches the zorvex technologies company knowledge base: leave, HR, medical, hiring, tech, incident"
                            "reports and travel expense policies. use it for any quesiton about the company policies or documents."
                            "returns the most relevant passages with their source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string to search in the knowledge base."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_city": get_city,
    "search_knowledge_base": search_knowledge_base
}

def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = {}
    
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"🛠️  {func_name}({func_args})")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP.get(func_name)
        result = fn(**func_args) if fn else None
    except Exception as e:
        result = f"Error executing function '{func_name}': {str(e)}"
    
    log(f"  ⤷ {func_name} returned: {result}")

    return str(result)