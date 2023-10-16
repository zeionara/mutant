import os
from os import environ as env
from shutil import rmtree

from click import group, argument, option

from transformers import pipeline
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, create_optimizer
from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from transformers.keras_callbacks import PushToHubCallback

from datasets import load_dataset


TOKEN = env['HUGGING_FACE_INFERENCE_API_TOKEN']


@group()
def main():
    pass


@main.command()
@argument('model', type = str)
@argument('output', type = str)
@option('--max-length', '-l', type = int, default = 1024)
@option('--batch-size', '-b', type = int, default = 4)
@option('--seed', '-s', type = int, default = 17)
@option('--epochs', '-e', type = int, default = 10)
@option('--pytorch', '-t', is_flag = True)
@option('--cached', '-c', is_flag = True, help = 'use cached dataset')
def fine_tune(model: str, output: str, max_length: int, batch_size: int, seed: int, epochs: int, pytorch: bool, cached: bool):
    # dataset_cache_path = f'assets/_baneks_{"pt" if pytorch else "tf"}'

    if not pytorch:
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable verbose logging from tensorflow

        import tensorflow as tf

    dataset_cache_path = 'assets/_baneks_tf'

    print(f'Fine-tuning model {model} {"using pytorch" if pytorch else "using tensorflow"} for {epochs} epochs. The result will be saved as {output}')

    # Prepare tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model

    model = AutoModelForCausalLM.from_pretrained(model) if pytorch else TFAutoModelForCausalLM.from_pretrained(model)

    # Prepare dataset

    dir_exists = os.path.isdir(dataset_cache_path)

    if pytorch or not cached or not dir_exists:
        if not pytorch and dir_exists:
            rmtree(dataset_cache_path)

        if not pytorch:
            if cached and not dir_exists:
                print('No preprocessed dataset found. Recomputing...')
            if not cached and dir_exists:
                print('Found cached dataset - will overwrite')

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

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False, return_tensors = 'pt' if pytorch else 'tf')

        if not pytorch:
            tf_dataset = model.prepare_tf_dataset(
                tokens,
                collate_fn = data_collator,
                shuffle = True,
                batch_size = batch_size
            )

            tf_dataset.save(dataset_cache_path)

    elif not pytorch:
        tf_dataset = tf.data.Dataset.load(dataset_cache_path)

    if pytorch:
        args = TrainingArguments(
            output_dir = output,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 8,
            num_train_epochs = epochs,
            weight_decay = 0.1,
            warmup_steps = 1_000,
            lr_scheduler_type = 'cosine',
            learning_rate = 5e-4,
            save_steps = 3_000,
            fp16 = True,
            push_to_hub = True
        )

        trainer = Trainer(
            model = model,
            tokenizer = tokenizer,
            args = args,
            data_collator = data_collator,
            train_dataset = tokens
        )

        trainer.train()
        trainer.push_to_hub()
    else:
        # print(tf_dataset)

        n_steps = len(tf_dataset)
        optimizer, _ = create_optimizer(
            init_lr = 5e-5,
            num_warmup_steps = 1_000,
            num_train_steps = n_steps,
            weight_decay_rate = 0.01
        )
        model.compile(optimizer = optimizer)

        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # print(f'Ready to train for {n_steps} steps')

        callback = PushToHubCallback(output_dir = output, tokenizer = tokenizer)

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
