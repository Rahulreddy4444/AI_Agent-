import os

START_SEQUENCE = "\033[90m" #ANSI escape code for gray text
END_SEQUENCE = "\033[0m"   #ANSI escape code to reset text color

def log(text):
    if os.environ.get("DEBUG", "False").lower() == "true":
        print(f"  {START_SEQUENCE}[DEBUG] {text}{END_SEQUENCE}")