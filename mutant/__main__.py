import os
from os import environ as env
import pickle

env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

import tensorflow as tf

from click import group, argument, option
from transformers import pipeline, TFGPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, create_optimizer
from transformers.keras_callbacks import PushToHubCallback
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
@option('--epochs', '-e', type = int, default = 10)
def fine_tune(model: str, max_length: int, batch_size: int, seed: int, epochs: int):
    dataset_cache_path = 'assets/_baneks_tf'

    print(f'Fine-tuning model {model}')

    # Prepare tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model

    config = AutoConfig.from_pretrained(
        model,
        vocab_size = len(tokenizer),
        n_ctx = max_length,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )
    model = TFGPT2LMHeadModel(config)

    # Prepare dataset

    if not os.path.isdir(dataset_cache_path):

        dataset = load_dataset('zeio/baneks')['train']  # .select(range(10))

        dataset = dataset.shuffle(seed = seed)

        def tokenize(item):
            outputs = tokenizer(
                item['text'],
                max_length = max_length,
                return_overflowing_tokens = True,
                truncation = True
            )

            return {'input_ids': outputs.input_ids}

        tokens = dataset.map(tokenize, batched = True, batch_size = batch_size, remove_columns = dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False, return_tensors = 'tf')

        tf_dataset = model.prepare_tf_dataset(
            tokens,
            collate_fn = data_collator,
            shuffle = True,
            batch_size = batch_size
        )

        tf_dataset.save(dataset_cache_path)

    else:
        tf_dataset = tf.data.Dataset.load(dataset_cache_path)

    print(tf_dataset)

    n_steps = len(tf_dataset)
    optimizer, _ = create_optimizer(
        init_lr = 5e-5,
        num_warmup_steps = 1_000,
        num_train_steps = n_steps,
        weight_decay_rate = 0.01
    )
    model.compile(optimizer = optimizer)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    print(f'Ready to train for {n_steps} steps')

    callback = PushToHubCallback(output_dir = 'fool', tokenizer = tokenizer)

    model.fit(tf_dataset, callbacks = [callback], epochs = epochs)


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
