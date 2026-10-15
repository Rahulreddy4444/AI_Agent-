import json
from openai import OpenAI
from  dotenv import load_dotenv
from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP
import os

load_dotenv() #by default it will look for a .env file in the current directory

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []
messages.append(
    {"role": "system", 
     "content": "You are a humble assistant and give response under 100 words."}
)

while True:

    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break

    messages.append(
        {"role": "user", 
         "content": user_input}
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        tools=TOOLS,
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments or "{}")

            fn = FUNCTION_NAME_TO_FUNCTION_MAP.get(func_name)
            result = fn(**func_args) if fn else None

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # 2nd call to the model after tool execution
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message
        

 
    messages.append(
        {"role": "assistant",
         "content": msg.content}
    )

    print("Assistant:", msg.content)