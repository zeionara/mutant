from os import environ as env

env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

from click import group, argument, option
from transformers import pipeline


TOKEN = env['HUGGING_FACE_INFERENCE_API_TOKEN']


@group()
def main():
    pass


@main.command()
@argument('text', type = str)
@option('--model', '-m', type = str, required = False)
@option('--max-length', '-l', type = int, required = False)
def run(text: str, model: str, max_length: int):
    # 1. Sentiment classification

    # classifier = pipeline('sentiment-analysis', token = TOKEN)

    # result = classifier(text)

    # print(result)

    # result = classifier([text, text])

    # print(result)

    # 2. Text generation

    # print(
    #     pipeline(
    #         task = 'text-generation',
    #         # model = 'mistralai/Mistral-7B-v0.1'
    #         model = model
    #     )(
    #         text,
    #         max_length = max_length
    #     )
    # )

    # 3. Mask filling

    print(
        '\n'.join([
            item['sequence'] for item in pipeline(
                task = 'fill-mask'
            )(
                text,
                top_k = 5
            )
        ])
    )


if __name__ == '__main__':
    main()
