## requirements
<pre>
pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes einops wandb
pip install evaluate rouge_score
</pre>
## commands: 
<pre>
Arguments:
-i --input_dir: dataset path
-m --mode: either 'train' or 'test'
-n --model_name: any autoregressive model from huggingface like gpt2
-c --ckpt_dir: the directory for saving checkpoints
</pre>

## train
<pre>
python main.py -i "daily_dialog/" -m "train" -n "gpt2" -c "res"
</pre>
## test
<pre>
python main.py -i "daily_dialog/" -m "test" -n "gpt2" -c "res"
</pre>

## structure
<pre>
GenerativeChatBot
     |
     ├── dailydialog
     │   ├── dialogues_topic.txt
     │   ├── readme.txt
     │   ├── test
     │   ├── train
     │   └── validation
     ├── evaluation.py: Eval script that calculates rouge, and bleu scores
     ├── main.py: The main script initiating the whole pipeline
     ├── parser.py: parse and preprocess dialy dialog and creates the dataset
     ├── report.docx
     ├── train.py: Training script that finetunes model with LORA
     └── res
         ├── ckpt_model
         │   ├── config.json
         │   └── generation_config.json
         └── ckpt_tokenizer
             ├── merges.txt
             ├── special_tokens_map.json
             ├── tokenizer.json
             ├── tokenizer_config.json
             └── vocab.json
</pre>
