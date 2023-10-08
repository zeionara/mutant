from os import environ as env

env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

from click import group, argument
from transformers import pipeline


TOKEN = env['HUGGING_FACE_INFERENCE_API_TOKEN']


@group()
def main():
    pass


@main.command()
@argument('text', type = str)
def run(text: str):
    # 1. Sentiment classification

    # classifier = pipeline('sentiment-analysis', token = TOKEN)

    # result = classifier(text)

    # print(result)

    # result = classifier([text, text])

    # print(result)

    # 2. Text generation

    print(
        pipeline('text-generation')(text)
    )


if __name__ == '__main__':
    main()
