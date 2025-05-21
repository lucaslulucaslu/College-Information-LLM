from google import genai
import os
from langfuse.decorators import observe, langfuse_context
from google.genai.types import GenerateContentConfig, ThinkingConfig

model = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@observe(as_type="generation")
def llm_wrapper(sys_prompt, user_prompt, response_format=None):
    if response_format:
        response = client.models.generate_content(
            model=model,
            contents=sys_prompt + user_prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_format,
                thinking_config=ThinkingConfig(thinking_budget=0)
            ),
        )
        langfuse_context.update_current_observation(
            model=model,
            output=response.parsed,
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
            },
        )
    else:
        response = client.models.generate_content(
            model=model,
            contents=sys_prompt + user_prompt,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(thinking_budget=0)
            ),
        )
        langfuse_context.update_current_observation(
            model=model,
            output=response.text,
            usage_details={
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
            },
        )
    return response


@observe(as_type="generation")
def llm_wrapper_streaming_trace(
    sys_prompt, user_prompt, full_response, input_tokens, output_tokens, total_tokens
):
    langfuse_context.update_current_observation(
        model=model,
        input={"system": sys_prompt, "user": user_prompt},
        output=full_response,
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
    )


def llm_wrapper_streaming(sys_prompt, user_prompt):
    response = client.models.generate_content_stream(
        model=model,
        contents=sys_prompt + user_prompt,
    )
    full_response = ""
    for chunk in response:
        full_response += chunk.text
        yield chunk.text
    input_tokens = chunk.usage_metadata.prompt_token_count
    output_tokens = chunk.usage_metadata.candidates_token_count
    total_tokens = chunk.usage_metadata.total_token_count
    llm_wrapper_streaming_trace(
        sys_prompt,
        user_prompt,
        full_response,
        input_tokens,
        output_tokens,
        total_tokens,
    )
