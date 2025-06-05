# llm_interface.py
import ollama

def query_ollama_model(model_name: str, prompt: str, system_message: str = None) -> str:
    """Queries a specified Ollama model."""
    try:
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})

        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error querying Ollama model {model_name}: {e}")
        return f"Error: Could not get response from {model_name}."

# Example usage (not part of the file, just for illustration)
# target_response = query_ollama_model("llama3", "What is the capital of France?")
# print(target_response)