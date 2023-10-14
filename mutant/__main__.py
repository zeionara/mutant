from os import environ as env

env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

import tensorflow as tf

from click import group, argument, option
from transformers import pipeline, TFGPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset


TOKEN = env['HUGGING_FACE_INFERENCE_API_TOKEN']


@group()
def main():
    pass


@main.command()
@argument('model', type = str, default = 'gpt2')
def fine_tune(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('zeio/baneks')['train'].select(range(10))

    dataset = dataset.shuffle(seed = 17)

    def tokenize(item):
        # return tokenizer(item['text'], return_tensors='tf', max_length = 1024, truncation = True, padding = True)
        outputs = tokenizer(
            item['text'],
            # return_tensors='tf',
            max_length = 100,
            return_overflowing_tokens = True,
            return_length = True,
            truncation = True
        )

        print(outputs['length'])

        return outputs

    # dataset.add_column('tokens', tokens)

    # tokens = dataset.map(tokenize, batched = True)

    # outputs = tokenizer(
    #     dataset[1]['text'],
    #     truncation = True,
    #     # return_tensors='tf',
    #     max_length = 100,
    #     return_overflowing_tokens = True,
    #     return_length = True
    # )

    # print(outputs['length'])

    tokens = dataset.map(tokenize)

    # outputs = tokenizer(
    #     dataset[:2]["text"],
    #     truncation=True,
    #     max_length=100,
    #     return_overflowing_tokens=True,
    #     return_length=True
    # )

    # print(f"Input IDs length: {len(outputs['input_ids'])}")
    # print(f"Input chunk lengths: {(outputs['length'])}")
    # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    # tokens.set_format('tf', columns = ('input_ids', 'attention_mask'))

    # tokens.batch(5)

    # for batch in tokens.iter(batch_size = 5):
    #     print(batch)

    # model = TFGPT2LMHeadModel.from_pretrained(model)

    # text = 'huggingface is the'

    # inputs = tokenizer(text, return_tensors = 'tf')
    # outputs = model(**inputs)

    # logits = outputs.logits[0, -1, :]
    # # print(outputs.logits.shape)
    # # argmax = tf.argmax(outputs.logits, axis = -1).numpy().tolist()[0]

    # # print(argmax)

    # # softmax = tf.math.softmax(logits, axis=-1)
    # argmax = tf.math.argmax(logits, axis=-1)

    # print(argmax)

    # print(text, "[", tokenizer.decode(argmax), "]")

    # # print(tokenizer.decode(outputs))


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

    print(
        pipeline(
            task = 'text-generation',
            # model = 'mistralai/Mistral-7B-v0.1'
            model = model
        )(
            text
            # max_length = max_length
        )
    )

    # 3. Mask filling

    # print(
    #     '\n'.join([
    #         item['sequence'] for item in pipeline(
    #             task = 'fill-mask'
    #         )(
    #             text,
    #             top_k = 5
    #         )
    #     ])
    # )


if __name__ == '__main__':
    main()
