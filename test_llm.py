"""
DataMind — Step 4: Test that the LLM API connection works.

This script does ONE thing: send a simple text prompt to Gemini and
print the response. If this works, we know the key is valid and the
library is installed correctly — then we can safely build on top of it
in main.py.

Run this directly with: python test_llm.py
(Not through uvicorn — this isn't part of the server, it's just a
quick standalone check.)
"""

import os

from dotenv import load_dotenv
from google import genai

# This line finds your .env file and loads whatever's inside it into
# the environment automatically — so you don't need to type "export"
# every single time you open a new terminal. It just quietly happens
# the moment this script runs.
load_dotenv()

# IMPORTANT: never type your actual API key directly into code like
# api_key="AI...". If you ever share this file, push it to GitHub, or
# even just take a screenshot, that key would leak. Instead, we read it
# from an "environment variable" — a value set outside the code, in
# your terminal session, that only your own machine knows.
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. "
        "See the instructions below this script for how to set it."
    )

# This creates a "client" — think of it as your connection handle to
# Google's servers, authenticated using your key.
client = genai.Client(api_key=API_KEY)

# This is the actual API call: send a prompt, get a response back.
# gemini-2.0-flash was retired — Google's own error message told us to
# switch to gemini-3.6-flash instead. This kind of thing (model names
# changing) happens often with AI APIs, so if this ever breaks again
# later, check the exact error message first — it usually tells you
# precisely what changed.
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly one sentence confirming you received this test message.",
)

print("SUCCESS! The model replied:")
print(response.text)