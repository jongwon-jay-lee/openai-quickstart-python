import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # supply your API key however you choose

promptsArray = ["Hello world, from", "How are you B", "I am fine. W", "The  fifth planet from the Sun is "]

stringifiedPromptsArray = json.dumps(promptsArray)

print(promptsArray)

prompts = [
    {
        "role": "user",
        "content": stringifiedPromptsArray
    }
]

batchInstruction = {
    "role": "system",
    "content": "Complete every element of the array. Reply with an array of all completions."
}

prompts.append(batchInstruction)
print("ChatGPT: ")
stringifiedBatchCompletion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                          messages=prompts,
                                                          max_tokens=1000)
batchCompletion = json.loads(stringifiedBatchCompletion.choices[0].message.content)
print(batchCompletion)
