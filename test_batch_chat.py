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


def stick_out(a, b):
    return a > b


def executable():
    # Mount Fuji is 3,776 meters tall.
    height_mount_fuji = 3776

    # The Sea of Japan is about 3,741 meters deep.
    depth_sea_of_japan = 3741

    # (Wrong inference)
    # Thus, the top of Mount Fuji would not stick out of the Sea of Japan.
    result = stick_out(height_mount_fuji, depth_sea_of_japan) is False

    # So the answer is no.
    return result
