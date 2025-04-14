from typing import Callable, List, TypeVar, Any
from promoai.general_utils.ai_providers import AIProviders
from promoai.prompting.prompt_engineering import ERROR_MESSAGE_FOR_MODEL_GENERATION
from promoai.general_utils import constants
from huggingface_hub import InferenceClient  # Add this import here

T = TypeVar('T')

def generate_result_with_error_handling(conversation: List[dict[str:str]],
                                        extraction_function: Callable[[str, Any], T],
                                        api_key: str,
                                        llm_name: str,
                                        ai_provider: str,
                                        max_iterations=5,
                                        additional_iterations=5,
                                        standard_error_message=ERROR_MESSAGE_FOR_MODEL_GENERATION) \
        -> tuple[str, any, list[Any]]:
    error_history = []
    for iteration in range(max_iterations + additional_iterations):
        if ai_provider == AIProviders.TOGETHER.value:
            response = generate_response_with_together(conversation, api_key, llm_name)
        else:
            # Handle other providers if needed
            raise Exception(f"AI provider {ai_provider} is not supported!")
            
        try:
            conversation.append({"role": "assistant", "content": response})
            auto_duplicate = iteration >= max_iterations
            code, result = extraction_function(response, auto_duplicate)
            return code, result, conversation  # Break loop if execution is successful
        except Exception as e:
            error_description = str(e)
            error_history.append(error_description)
            if constants.ENABLE_PRINTS:
                print("Error detected in iteration " + str(iteration + 1))
            new_message = f"Executing your code led to an error! " + standard_error_message + f" This is the error message: {error_description}"
            conversation.append({"role": "user", "content": new_message})

    raise Exception(llm_name + " failed to fix the errors after " + str(max_iterations + 5) +
                    " iterations! This is the error history: " + str(error_history))

def print_conversation(conversation):
    if constants.ENABLE_PRINTS:
        print("\n\n")
        for index, msg in enumerate(conversation):
            print("\t%d: %s" % (index, str(msg).replace("\n", " ").replace("\r", " ")))
        print("\n\n")

def generate_response_with_history_ollama(conversation_history, api_key, model_name) -> str:
    """
    Generates a response from Ollama using the conversation history.
    
    :param conversation_history: The conversation history to be included
    :param api_key: Not used for Ollama (can be None or empty)
    :param model_name: Ollama model name to use (e.g., 'gemma:2b')
    :return: The content of the LLM response
    """
    import requests
    
    messages = [
        {
            "role": message["role"],
            "content": message["content"]
        } for message in conversation_history
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]
    except Exception as e:
        raise Exception(f"Ollama connection failed: {str(e)}")
    
    
from huggingface_hub import InferenceClient

def generate_response_with_together(conversation_history, api_key, model_name) -> str:
    """
    Generates a response using Together AI via Hugging Face InferenceClient.
    """
    try:
        # Initialize the client
        client = InferenceClient(
            provider="together",
            api_key=api_key,
        )
        
        # Format messages for the API
        messages = []
        for message in conversation_history:
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        # Create completion
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,  # Low for deterministic outputs
        )
        
        # Return the response
        return completion.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Together API connection failed: {str(e)}")