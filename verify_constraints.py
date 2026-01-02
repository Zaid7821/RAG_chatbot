import os
from dotenv import load_dotenv
from app import build_prompt_with_context, call_groq_chat

# Load environment variables
load_dotenv()

def test_prompt_construction():
    print("Testing Prompt Construction...")
    question = "What is the capital of France?"
    contexts = ["Paris is the capital of France."]
    prompt = build_prompt_with_context(question, contexts)
    print(f"Generated Prompt:\n{prompt}\n")
    assert "Strictly based on the provided context" in prompt or "STRICTLY based on the provided context" in prompt
    assert "Do not use any outside knowledge" in prompt

def test_llm_response_strictness():
    print("Testing LLM Response Strictness...")
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping LLM test: GROQ_API_KEY not found.")
        return

    # Test case 1: Answer in context
    question = "What is the capital of France?"
    contexts = ["Paris is the capital of France."]
    prompt = build_prompt_with_context(question, contexts)
    response = call_groq_chat(
        system="You are a strict assistant that only answers based on the provided text.",
        user=prompt
    )
    print(f"Q: {question}\nContext: {contexts}\nA: {response}\n")

    # Test case 2: Answer NOT in context
    question = "What is the capital of Germany?"
    contexts = ["Paris is the capital of France."]
    prompt = build_prompt_with_context(question, contexts)
    response = call_groq_chat(
        system="You are a strict assistant that only answers based on the provided text.",
        user=prompt
    )
    print(f"Q: {question}\nContext: {contexts}\nA: {response}\n")

if __name__ == "__main__":
    test_prompt_construction()
    test_llm_response_strictness()
