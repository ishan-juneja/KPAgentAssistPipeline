import time
from openai import OpenAI

def prompt_llm(prompts, system_prompt=None, examples=None):
    """
    Prompts an LLM with multiple user messages as context,
    and optionally few-shot examples.

    Args:
        prompts (list of str): Each item is a user message in order.
        system_prompt (str or None): Optional system prompt.
        examples (list of (str, str)): Optional list of (user, assistant) example pairs.
    
    Returns:
        str: The model's final response text.
    """

    

        
    client = OpenAI(
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

    model = "qwen/qwen2.5-vl-72b-instruct:free"

    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Add examples as prior turns
    if examples:
        for user_text, assistant_text in examples:
            messages.append({
                "role": "user",
                "content": user_text
            })
            messages.append({
                "role": "assistant",
                "content": assistant_text
            })

    # Add real prompts
    for prompt_text in prompts:
        messages.append({
            "role": "user",
            "content": prompt_text
        })

    attempt = 1
    wait_time = 3
    while(True):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Having the following error: ", e)
            print("Will Re Run again!")
            time.sleep(wait_time * (attempt * 0.5))
            attempt +=1

    