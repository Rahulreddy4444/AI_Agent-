from datetime import datetime
import json
import requests

from Utilities.utilities import log

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city: str):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text if response.status_code == 200 else "Could not retrieve weather information."

def get_city():
    return "Bangalore"

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
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_city": get_city
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