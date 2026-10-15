
from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
}