import dataclasses
import os
import string
import random
import sys
import math
import logging
import time
import tqdm
import copy
import re
import openai
from openai import openai_object
from typing import Optional, Sequence, Union

openai.api_key = os.getenv("OPENAI_API_KEY")
StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # TODO: gpt-4 cannot handle args below
    # suffix: Optional[str] = None
    # logprobs: Optional[int] = None
    # echo: bool = False


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./ko_prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)

    if model_name == "text-davinci-003":
        prompt_batches = [
            prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
            for batch_id in range(int(math.ceil(num_prompts / batch_size)))
        ]
    elif model_name == "gpt-4" or model_name == 'gpt-3.5-turbo':
        prompt_batches = []
        for batch_id in range(int(num_prompts / batch_size)):
            curr_batch = []
            for d_idx in range(batch_size):
                curr_batch.append({
                    "role": "system",
                    "content": prompts[batch_id + d_idx]
                })
            prompt_batches.append(curr_batch)
    else:
        raise ValueError(f"check the model_name: {model_name}")

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                if model_name == "text-davinci-003":
                    completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                elif model_name == "gpt-4" or model_name == 'gpt-3.5-turbo':
                    completion_batch = openai.ChatCompletion.create(messages=prompt_batch, model=model_name)
                else:
                    raise ValueError(f"check the model_name: {model_name}")

                choices = completion_batch.choices
                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def post_process_gpt4_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["message"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def main():
    temperature = 1.0
    top_p = 1.0
    request_batch_size = 2
    num_prompt_instructions = 1
    # model_name = "text-davinci-003"   # max_tokens = 4097
    # model_name = "gpt-4"    # max_token = 8192
    # model_name = "gpt-4-32k"  # max_token = 32768
    model_name = 'gpt-3.5-turbo'

    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=1000,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        stop=["\n20", "20.", "20."],
    )
    seed_instruction_data = [
        {"instruction": "아침 식사로 계란은 안 들어가지만 단백질이 들어있고 열량이 700~1000칼로리 정도 되는 것이 있습니까?", "input": "", "output": "네, 오트밀 바나나 단백질 쉐이크 1개와 베이컨 4조각을 드실 수 있습니다. 오트밀 바나나 단백질 셰이크는 오트밀 1/2컵, 유청 단백질 분말 60g, 중간 바나나 1/2개, 아마씨 오일 1tbsp, 물 1/2컵으로 총 550칼로리를 포함할 수 있습니다. 베이컨 4조각은 약 200칼로리를 포함하고 있습니다."},
        {"instruction": "주말에 볼 만한 영화를 추천하고 그 이유를 설명해 주세요.", "input": "", "output": "감동과 감동을 동시에 주는 훌륭한 영화이기 때문에 저는 \"쇼생크 구원\"이라는 영화를 추천하고 싶습니다. 억울하게 감옥에 갇힌 한 남자와 희망과 존엄성을 유지하기 위한 몸부림 이야기입니다. 그것은 인간의 회복력과 희망에 대한 능력에 대해 생각하게 해줄 것이기 때문에 주말에 보기 좋은 영화입니다."},
        {"instruction": "당신의 생각에, 당신은 그 분야에서 경력이 없는 사람들과 주어진 주제의 관계를 어떻게 설명할 것입니까?", "input": "기계 학습 및 데이터 마이닝", "output": "그것들은 데이터에서 정보를 추출하는 것을 다루는 매우 밀접하게 관련된 두 가지 연구 분야입니다. 머신 러닝은 데이터에서 학습하고 새로운 데이터에 대한 예측을 할 수 있는 알고리즘의 개발과 관련이 있습니다. 데이터 마이닝은 데이터에서 패턴과 추세를 추출하는 것과 관련이 있습니다. 두 분야 모두 통계, 인공지능 및 패턴 인식을 포함한 다양한 기술을 사용합니다."},
        {"instruction": "사실을 고려할 때, 그것들을 일관성 있고 매혹적인 이야기로 구성하세요.", "input": "사실:\n- Person X는 남아프리카에서 태어났습니다.\n- Person X는 그의 행복한 어린 시절의 대부분을 친구들과 놀면서 보냈습니다.\n- Person X는 대학 이후에야 프로그래밍을 배우기 시작했습니다.\n- Person X는 억만장자가 되었습니다.", "output": "기업가 Person X는 국제적인 회사의 CEO로서 큰 명성을 얻었습니다. 그의 회사인 돌핀은 2022년 현재 미국에서 가장 수익성이 높은 핀테크 기업 중 하나입니다. 핀테크 분야에서 성공했음에도 불구하고, Person X는 대학 이전에 프로그래밍을 배우기 시작하지 않았습니다. 남아프리카의 중산층 가정에서 태어난 Person X는 어린 시절 대부분을 컴퓨터로 노는 대신 친구들과 어울리며 보냈습니다. \"아버지가 컴퓨터를 사주지 않으셔서 그때는 프로그래밍에 대해 전혀 몰랐습니다.\"라고 X 씨가 한 인터뷰에서 말했습니다. 하지만, 그는 대학에 간 후에 프로그래밍에 대한 그의 재능을 정말로 보여주었습니다. 그는 2년차에 첫 스타트업을 시작했고, 그 스타트업은 나중에 마이크로소프트에 인수됩니다."},
    ]

    batch_inputs = []

    for _ in range(request_batch_size):
        # only sampling from the seed tasks
        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        prompt = encode_prompt(prompt_instructions)
        batch_inputs.append(prompt)

    results = openai_completion(
        prompts=batch_inputs,
        model_name=model_name,
        batch_size=request_batch_size,
        decoding_args=decoding_args,
        logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    )

    instruction_data = []
    for result in results:
        new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
        instruction_data += new_instructions

    print(instruction_data)


if __name__ == "__main__":
    main()

