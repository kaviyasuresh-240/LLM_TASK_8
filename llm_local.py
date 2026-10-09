import pandas as pd
import requests
from datetime import datetime

# Load prompts
prompts = pd.read_csv("dataset/prompts.csv")

# Use Gemma 2B model you pulled
model = "gemma:2b"
responses = []
history = []

for index, row in prompts.iterrows():
    prompt = row["prompt"]
    print(f"Processing prompt {index + 1}/{len(prompts)}...")

    # Call Ollama REST API (disable streaming for clean JSON)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    # Safely decode JSON
    try:
        data = response.json()
        answer = data["message"]["content"]
    except ValueError:
        print(f"JSON decode error for prompt {index}: {response.text}")
        answer = "Error decoding response"

    # Collect structured response
    responses.append({
        "prompt_id": row["prompt_id"],
        "domain": row["domain"],
        "prompt": prompt,
        "response": answer
    })

    # Log conversation history
    history.append(f"[{datetime.now()}] USER: {prompt}\nMODEL: {answer}\n\n")

# Save responses
result = pd.DataFrame(responses)
result.to_csv("dataset/llm_responses.csv", index=False)

# Save conversation history
with open("dataset/conversation_history.txt", "w", encoding="utf-8") as f:
    f.writelines(history)

# Create comparison file (can later merge with other models)
comparison = result[["prompt_id", "domain", "prompt", "response"]]
comparison.to_csv("dataset/response_comparison.csv", index=False)

# Add observations file (template for analysis)
with open("dataset/observations.txt", "w", encoding="utf-8") as f:
    f.write("Observations on Gemma 2B responses:\n")
    f.write("- Note how different domains affect answer quality.\n")
    f.write("- Compare with other models later for accuracy.\n")

print("All files saved successfully: llm_responses.csv, response_comparison.csv, conversation_history.txt, observations.txt")
