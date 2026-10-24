# standard library imports
import json
import os
from types import SimpleNamespace

# third-party imports
import requests
from openai import OpenAI
from  dotenv import load_dotenv

# local module imports
from tools import TOOLS, run_tool
from Utilities.utilities import log

load_dotenv() #by default it will look for a .env file in the current directory

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []
messages.append(
    {"role": "system", 
     "content": "You are a general-purpose assistant. Answer questions about programming and other topics directly. Use the available tools only when a request needs current time, weather, or city information. Do not claim your capabilities are limited to those tools."}
)

# safety limit to avoid infinite loops in case of tool-calling
MAX_STEPS = 5

def stream_chat(history): 
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True
    )

    full_response = ""
    calls = {}   # index -> tool call being built
    label_printed = False

    # streaming started
    try:
        for chunk in stream:
            if not chunk.choices:                       # some servers send chunks with no choices
                continue
            delta = chunk.choices[0].delta

            if not label_printed:
                print("Assistant:", end=" ", flush=True)
                label_printed = True

            # 1. streaming the assistant's response
            piece = delta.content
            if piece:
                print(piece, end="", flush=True) #streaming
                full_response += piece

            # 2. handling tool calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments

    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
        calls = {}  # Reset calls if interrupted    

    if label_printed:
        print() # for newline after the assistant's response

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response, calls_list

def build_tool_calls_payload(tool_calls):
    """
    converts a list of tool call objects into a payload format suitable for sending to the model.
    Each tool call object is expected to have an 'id', 'function.name', and 'function.arguments' attribute.
    The function returns a list of dictionaries, where each dictionary represents a tool call with its id and function details.
    """
    payload = []
    for tool_call in tool_calls:
        payload.append({
            "id": tool_call.id,
            "function": {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
        })
    return payload

while True:

    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chat. Goodbye!")
        break

    messages.append(
        {"role": "user", 
         "content": user_input}
    )

    #chain tool-calling 
    for step in range(MAX_STEPS):
        full_response, tool_calls = stream_chat(messages)
        
        # msg = response.choices[0].message

        if not tool_calls:
            break

        log(f"Step {step + 1}: model requested {len(tool_calls)} tool call(s).")

        # messages.append(msg)
        messages.append(
            {"role": "assistant",
            "content": full_response,
            "tool_calls": build_tool_calls_payload(tool_calls)}
        )

        for tool_call in tool_calls:
            result = run_tool(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    else:
        fallback_response = "Reached maximum steps without a final response. Please try again."
        messages.append({
            "role": "assistant", 
            "content": fallback_response})
        print("Assistant:", fallback_response)
        continue
