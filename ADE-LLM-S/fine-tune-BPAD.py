import os, sys
import random

import numpy as np

from prompt import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params

batch_size = 16
micro_batch_size = 8
gradient_accumulation_steps = batch_size // micro_batch_size

# device = torch.device("cuda:1")

llm_path = "meta-llama/Llama-3.1-8B-Instruct"

data_path='dataset/process_model/train_dataset.jsonl'

data_name = os.path.splitext(os.path.basename(data_path))[0]

check_point_save_path = 'checkpoints'
ft_model_name = "llama-fine-tuned"


checkpoint_path = None
#checkpoint_path = f'{check_point_save_path}/checkpoint-203000'

max_length = 512


tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map='auto')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, device_map="auto")

model = prepare_model_for_kbit_training(model)
'''
- r, the dim of the low_rank matrices
- lora_alpha, scaling factor, the weight is scaled by lora_alpha/r, 
  the higher value assigns more weight to the LoRA activations
- target_modules: default is "q_proj", "v_proj"
- bias, the recommend setting bias to None first, and then lora_only, before trying all.
'''
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

### compare trainable parameters
peft_p = print_number_of_trainable_model_parameters(model)


def generate_prompt(trace, label=None, cause=None, all=True, prompt_template=prompt_template):
    p1 = preface1[random.randint(0, len(preface1) - 1)]
    p2 = preface2[random.randint(0, len(preface2) - 1)]
    ask_c = ask_cause[random.randint(0, len(ask_cause) - 1)]

    if cause:
        if all:
            res = prompt_template["prompt_with_cause_all"].format(p1=p1, p2=p2, ask_c=ask_c,
                                                                  trace=trace, label=label, cause=cause)
        else:
            res = prompt_template["prompt_with_cause_q"].format(p1=p1, p2=p2, ask_c=ask_c,
                                                                trace=trace, label=label)
    else:
        if all:
            res = prompt_template["prompt_no_cause_all"].format(p1=p1, p2=p2,
                                                                trace=trace, label=label)
        else:
            res = prompt_template["prompt_no_cause_q"].format(p1=p1, p2=p2,
                                                              trace=trace)
    return res

def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(batch):
    full_prompts = []
    user_prompt_lens = []

    for trace, label, cause in zip(batch["trace"], batch["label"], batch["cause"]):
        full_prompt = generate_prompt(trace, label, cause)
        user_prompt = generate_prompt(trace, label, cause, all=False)

        full_prompts.append(full_prompt)
        user_prompt_len = len(tokenizer(user_prompt, truncation=True, max_length=max_length)["input_ids"])
        user_prompt_lens.append(user_prompt_len)

    # Tokenize full prompts in batch
    tokenized = tokenizer(
        full_prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        user_len = user_prompt_lens[i]
        label_ids = input_ids.clone()
        label_ids[:user_len] = -100  # Mask user prompt part
        labels.append(label_ids)

    tokenized["labels"] = torch.stack(labels)
    return {k: v.numpy() for k, v in tokenized.items()}


if os.path.exists(f'tokenized_{data_name}.jsonl'):
    print(f"Loading data from 'tokenized_{data_name}.jsonl'")
    dataset = datasets.load_dataset("json",
                                    data_files=f'tokenized_{data_name}.jsonl',cache_dir="/gpfs/work5/0/prjs1629/hf_cache")
    train_data = dataset["train"]
    #train_data = dataset["train"].select(range(2))  # <-- only 2 samples

else:
    print(f"Loading data from '{data_path}'")
    dataset = datasets.load_dataset("json",
                                    data_files=data_path, cache_dir="/gpfs/work5/0/prjs1629/hf_cache")

    #dataset = dataset['train']
    #dataset = dataset.train_test_split(train_size=1000, test_size=1000, shuffle=True, seed=42)
    #dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    cols = ["trace", "label", "cause"]
    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols, batched=True, batch_size=64, num_proc=8)
    # val_data = dataset["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols, )
    print(f"Saving tokenized data to 'tokenized_{data_name}.jsonl'")
    train_data.to_json(f'tokenized_{data_name}.jsonl')



import transformers

##############train
transformers.logging.set_verbosity_info()

args = TrainingArguments(
    output_dir=check_point_save_path,
    num_train_epochs=1,
    # max_steps=1,
    fp16=True,
    optim="paged_adamw_32bit",
    learning_rate=5e-5,
    lr_scheduler_type="polynomial",
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': True},
    group_by_length=True,
    # logging_steps=10,
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    disable_tqdm=False
    # evaluation_strategy ='steps',
    # eval_steps=2,
    # dataloader_num_workers = 2,
    # dataloader_prefetch_factor=2
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    # eval_dataset=val_data,
    args=args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

# silence the warnings. re-enable for inference!
model.config.use_cache = False

if checkpoint_path is not None:
    trainer.train(checkpoint_path)
else:
    trainer.train()
model.save_pretrained(ft_model_name)
print('model train is finished')

