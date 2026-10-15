from openai import OpenAI
from  dotenv import load_dotenv
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
        messages=messages
    )

    messages.append(
        {"role": response.choices[0].message.role,
         "content": response.choices[0].message.content}
    )

    print("Assistant:", response.choices[0].message.content)