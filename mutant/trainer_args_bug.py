import tensorflow as tf

from transformers import TrainingArguments

print('Listing tf gpus...')

print(tf.config.list_physical_devices("GPU"))

print('Initializing training arguments...')

args = TrainingArguments(output_dir="output")

print('Initialization has completed')
