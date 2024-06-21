import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import deepspeed
# from huggingface_hub import login
# login()
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] =  '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'
print("I am done with hf things")
# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset['train'] = dataset['train'].select(range(20))
dataset['validation'] = dataset['validation'].select(range(20))
dataset['test'] = dataset['test'].select(range(20))
print("Dataset loading done")

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name,padding = "max_length", max_length = 512, truncation = True, cache_dir = '/scratch/bbzy/neeraja1504/')
model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir = '/scratch/bbzy/neeraja1504/')
tokenizer.pad_token = tokenizer.eos_token
print("Model loading done")
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets['train'][0])

tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,
#     deepspeed="./ds_config.json"
# )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling, so no masked language modeling
)

training_args = TrainingArguments(output_dir='personal/Neeraja/results', num_train_epochs=1, logging_steps=100, save_strategy='no',
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='personal/Neeraja/logs', fp16=True, deepspeed='./ds_config.json')

print("Data collator check:", data_collator([tokenized_datasets['train'][0], tokenized_datasets['train'][1]]))

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator= data_collator
)

# Train the model
trainer.train()


#   // "fp16": {
#   //   "enabled": true,
#   //   "min_loss_scale": 1,
#   //   "opt_level": "O3"
#   // },