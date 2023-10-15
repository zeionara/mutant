import os
from os import environ as env
import pickle

env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

import tensorflow as tf

from click import group, argument, option
from transformers import pipeline, TFGPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_dataset


TOKEN = env['HUGGING_FACE_INFERENCE_API_TOKEN']


@group()
def main():
    pass


@main.command()
@argument('model', type = str, default = 'gpt2')
@option('--max-length', '-l', type = int, default = 1024)
@option('--batch-size', '-b', type = int, default = 128)
@option('--seed', '-s', type = int, default = 17)
def fine_tune(model: str, max_length: int, batch_size: int, seed: int):
    dataset_cache_path = 'assets/_baneks_tf'

    # Prepare tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model

    config = AutoConfig.from_pretrained(
        'gpt2',
        vocab_size = len(tokenizer),
        n_ctx = max_length,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    model = TFGPT2LMHeadModel(config)

    # Prepare dataset

    if not os.path.isfile(dataset_cache_path):

        dataset = load_dataset('zeio/baneks')['train']  # .select(range(10))

        dataset = dataset.shuffle(seed = seed)

        def tokenize(item):
            # print(item)
            # return tokenizer(item['text'], return_tensors='tf', max_length = 1024, truncation = True, padding = True)
            outputs = tokenizer(
                item['text'],
                # return_tensors='tf',
                max_length = max_length,
                return_overflowing_tokens = True,
                # return_length = True,
                # padding = True,
                truncation = True
            )

            # print(outputs['length'])

            return {'input_ids': outputs.input_ids}
            # return outputs

            # print(tokenizer.decode(outputs.input_ids[-1]))

            # print(outputs['overflow_to_sample_mapping'])

            # return outputs

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

        tokens = dataset.map(tokenize, batched = True, batch_size = batch_size, remove_columns = dataset.column_names)

        # model(model.dummy_inputs)
        # model.summary()

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False, return_tensors = 'tf')

        # batch = data_collator([tokens[i] for i in range(5)])
        # for key in batch:
        #     print(f'{key} shape: {batch[key].shape}')

        tf_train_dataset = model.prepare_tf_dataset(
            tokens,
            collate_fn = data_collator,
            shuffle = True,
            batch_size = batch_size
        )

        tf_train_dataset.save(dataset_cache_path)

        # with open(dataset_cache_path, 'wb') as file:
        #     pickle.dump(tf_train_dataset, file)

    else:
        tf_train_dataset = tf.data.Dataset.load(dataset_cache_path)

        # with open(dataset_cache_path, 'rb') as file:
        #     tf_train_dataset = pickle.load(file)

    print(tf_train_dataset)

    # model_size = sum(t.numel() for t in model.parameters())
    # print(f'Model size: {model.num_parameters()/1000**2:.1f}M parameters')

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
