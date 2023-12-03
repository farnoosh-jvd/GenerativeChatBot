from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer


def creat_peft_cfg():


  #Create PEFT cfg
  lora_alpha = 16
  lora_dropout = 0.1
  lora_r = 64

  peft_config = LoraConfig(
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      r=lora_r,
      bias="none",
      task_type="CAUSAL_LM"
  )

  return peft_config


def set_training_args(out_dir):

  per_device_train_batch_size = 4
  gradient_accumulation_steps = 4
  optim = "paged_adamw_32bit"
  save_steps = 0.25
  logging_steps = 1000
  learning_rate = 1e-4
  num_train_epochs = 50
  warmup_ratio = 0.03
  max_grad_norm = 0.7
  lr_scheduler_type = "constant"

  # set training args to be passed to trainer
  training_arguments = TrainingArguments(
      output_dir=out_dir,
      per_device_train_batch_size=per_device_train_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      optim=optim,
      save_steps=save_steps,
      logging_steps=logging_steps,
      learning_rate=learning_rate,
      fp16=True,
      max_grad_norm=max_grad_norm,
      num_train_epochs=num_train_epochs,
      warmup_ratio=warmup_ratio,
      group_by_length=True,
      lr_scheduler_type=lr_scheduler_type,
  )

  return training_arguments

def train(model, dataset, tokenizer, out_dir):

  max_seq_length = 512
  peft_config = creat_peft_cfg()
  training_arguments = set_training_args(out_dir)

  #Create trainer

  trainer = SFTTrainer(
      model=model,
      train_dataset=dataset,
      peft_config=peft_config,
      dataset_text_field="text",
      max_seq_length=max_seq_length,
      tokenizer=tokenizer,
      args=training_arguments,
  )

  trainer.train()
  
  return model
