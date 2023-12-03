#requirements

pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes einops wandb
pip install evaluate rouge_score

#commands: 
"""
Arguments:
-i --input_dir: dataset path
-m --mode: either 'train' or 'test'
-n --model_name: any autoregressive model from huggingface like gpt2
-c --ckpt_dir: the directory for saving checkpoints
"""

## train
python main.py -i "daily_dialog/" -m "train" -n "gpt2" -c "res"

## test
python main.py -i "daily_dialog/" -m "test" -n "gpt2" -c "res"


#structure
LG_takehome
  |__daily_dialog
  |    |_test
  |    |_train
  |    |_validation
  |    |_*.txt
  |
  |__res
  |
  |__parser.py: parse and preprocess dialy dialog and creates the dataset
  |__main.py: The main script initiating the whole pipeline
  |__train.py: Training script that finetunes model with LORA
  |__evaluation.py: Eval script that calculates rouge, and bleu scores
  

