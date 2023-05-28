import openai
import json
import os
import random
from depr_test_inst_batch_chat import init_prompt, encode_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")  # supply your API key however you choose


seed_instruction_data = [
    {"instruction": "아침 식사로 계란은 안 들어가지만 단백질이 들어있고 열량이 700~1000칼로리 정도 되는 것이 있습니까?", "input": "", "output": "네, 오트밀 바나나 단백질 쉐이크 1개와 베이컨 4조각을 드실 수 있습니다. 오트밀 바나나 단백질 셰이크는 오트밀 1/2컵, 유청 단백질 분말 60g, 중간 바나나 1/2개, 아마씨 오일 1tbsp, 물 1/2컵으로 총 550칼로리를 포함할 수 있습니다. 베이컨 4조각은 약 200칼로리를 포함하고 있습니다."},
    {"instruction": "주말에 볼 만한 영화를 추천하고 그 이유를 설명해 주세요.", "input": "", "output": "감동과 감동을 동시에 주는 훌륭한 영화이기 때문에 저는 \"쇼생크 구원\"이라는 영화를 추천하고 싶습니다. 억울하게 감옥에 갇힌 한 남자와 희망과 존엄성을 유지하기 위한 몸부림 이야기입니다. 그것은 인간의 회복력과 희망에 대한 능력에 대해 생각하게 해줄 것이기 때문에 주말에 보기 좋은 영화입니다."},
    {"instruction": "당신의 생각에, 당신은 그 분야에서 경력이 없는 사람들과 주어진 주제의 관계를 어떻게 설명할 것입니까?", "input": "기계 학습 및 데이터 마이닝", "output": "그것들은 데이터에서 정보를 추출하는 것을 다루는 매우 밀접하게 관련된 두 가지 연구 분야입니다. 머신 러닝은 데이터에서 학습하고 새로운 데이터에 대한 예측을 할 수 있는 알고리즘의 개발과 관련이 있습니다. 데이터 마이닝은 데이터에서 패턴과 추세를 추출하는 것과 관련이 있습니다. 두 분야 모두 통계, 인공지능 및 패턴 인식을 포함한 다양한 기술을 사용합니다."},
    {"instruction": "사실을 고려할 때, 그것들을 일관성 있고 매혹적인 이야기로 구성하세요.", "input": "사실:\n- Person X는 남아프리카에서 태어났습니다.\n- Person X는 그의 행복한 어린 시절의 대부분을 친구들과 놀면서 보냈습니다.\n- Person X는 대학 이후에야 프로그래밍을 배우기 시작했습니다.\n- Person X는 억만장자가 되었습니다.", "output": "기업가 Person X는 국제적인 회사의 CEO로서 큰 명성을 얻었습니다. 그의 회사인 돌핀은 2022년 현재 미국에서 가장 수익성이 높은 핀테크 기업 중 하나입니다. 핀테크 분야에서 성공했음에도 불구하고, Person X는 대학 이전에 프로그래밍을 배우기 시작하지 않았습니다. 남아프리카의 중산층 가정에서 태어난 Person X는 어린 시절 대부분을 컴퓨터로 노는 대신 친구들과 어울리며 보냈습니다. \"아버지가 컴퓨터를 사주지 않으셔서 그때는 프로그래밍에 대해 전혀 몰랐습니다.\"라고 X 씨가 한 인터뷰에서 말했습니다. 하지만, 그는 대학에 간 후에 프로그래밍에 대한 그의 재능을 정말로 보여주었습니다. 그는 2년차에 첫 스타트업을 시작했고, 그 스타트업은 나중에 마이크로소프트에 인수됩니다."},
]


# promptsArray = ["Hello world, from", "How are you B", "I am fine. W", "The  fifth planet from the Sun is "]
batch_inputs = []
num_prompt_instructions = 3
request_batch_size = 2

for _ in range(request_batch_size):
    prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
    prompt = encode_prompt(prompt_instructions)
    batch_inputs.append(prompt)
    print(prompt)
    print("---" * 50)
    print()

stringifiedPromptsArray = json.dumps(batch_inputs, ensure_ascii=False)

prompts = [
    {
        "role": "user",
        "content": stringifiedPromptsArray
    }
]

batchInstruction = {
    "role": "system",
    "content": init_prompt(),
}

prompts.append(batchInstruction)
print("ChatGPT: ")
stringifiedBatchCompletion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                          messages=prompts,
                                                          max_tokens=1000)
print(stringifiedBatchCompletion)
batchCompletion = stringifiedBatchCompletion.choices[0].message.content
# TODO: gpt-3.5 above doesn't support batch?!
print(batchCompletion)
