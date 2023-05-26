import openai
import os


def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': '아침 식사로 계란은 안 들어가지만 단백질이 들어있고 열량이 700~1000칼로리 정도 되는 것이 있습니까?'}
        ],
        temperature=0
    )

    print(completion['choices'][0]['message']['content'])


if __name__ == "__main__":
    main()