import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def build_prompt_with_context(question: str, contexts: list) -> str:
    # Copy of the function from app.py
    header = (
        "You are a strict study assistant. You must answer the question STRICTLY based on the provided context below. "
        "Do not use any outside knowledge, general information, or pre-training data. "
        "If the answer cannot be found in the context, you MUST say exactly: 'I cannot answer this based on the provided documents.'\n\n"
    )
    ctxs = "\n\n".join([f"Context {i+1}:\n{c}" for i,c in enumerate(contexts)])
    prompt = f"{header}{ctxs}\n\nUser question: {question}\n\nAnswer concisely based ONLY on the context above. Do not hallucinate."
    return prompt

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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Skipping LLM test: GROQ_API_KEY not found.")
        return

    client = Groq(api_key=api_key)
    model_name = "openai/gpt-oss-20b" # Using the same model name as in app.py if possible, or a valid one. 
    # app.py uses "openai/gpt-oss-20b" which seems to be a placeholder or custom model. 
    # I should check what models are available or use a standard one like "llama3-70b-8192" or "mixtral-8x7b-32768" if that one fails.
    # But let's try the one in app.py first.
    
    # Actually, "openai/gpt-oss-20b" looks suspicious for Groq. Groq usually supports llama, mixtral, gemma.
    # Let's check app.py again. Line 19: MODEL_NAME = "openai/gpt-oss-20b"
    # If that works in the app, it should work here.

    def call_chat(system, user):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.0,
                max_tokens=600
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

    with open("verification_results.txt", "w") as f:
        # Test case 1: Answer in context
        f.write("--- Test Case 1: Answer in context ---\n")
        question = "What is the capital of France?"
        contexts = ["Paris is the capital of France."]
        prompt = build_prompt_with_context(question, contexts)
        response = call_chat(
            system="You are a strict assistant that only answers based on the provided text.",
            user=prompt
        )
        f.write(f"Q: {question}\nContext: {contexts}\nA: {response}\n\n")

        # Test case 2: Answer NOT in context
        f.write("--- Test Case 2: Answer NOT in context ---\n")
        question = "What is the capital of Germany?"
        contexts = ["Paris is the capital of France."]
        prompt = build_prompt_with_context(question, contexts)
        response = call_chat(
            system="You are a strict assistant that only answers based on the provided text.",
            user=prompt
        )
        f.write(f"Q: {question}\nContext: {contexts}\nA: {response}\n\n")


if __name__ == "__main__":
    test_prompt_construction()
    test_llm_response_strictness()
