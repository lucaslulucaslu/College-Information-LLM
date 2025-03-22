from google import genai
import os

model = "gemini-2.0-flash"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def llm_wrapper_streaming(sys_prompt, user_prompt):
    response = client.models.generate_content_stream(
        model=model,
        contents=sys_prompt + user_prompt,
    )
    for chunk in response:
        yield chunk.text

def llm_wrapper(sys_prompt, user_prompt, response_format = None):
    if response_format:
        response= client.models.generate_content(
            model=model,
            contents=sys_prompt + user_prompt,
            config = {
                "response_mime_type": "application/json",
                "response_schema": response_format
            }
        )
    else:
        response= client.models.generate_content(
            model=model,
            contents=sys_prompt + user_prompt
        )
    return response