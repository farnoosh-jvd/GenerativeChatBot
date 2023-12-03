import os, sys, getopt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser import parse_data
from train import train
from evaluation import eval_metrics

def main(argv):

  in_dir = ''
  mode = ''
  model_name = ''
  ckpt_dir = ''

  try:
      opts, args = getopt.getopt(argv,"h:i:m:n:c:",["in_dir=", "mode=", "model_name=", "ckpt_dir="])
  except getopt.GetoptError:
      print("python3 main.py -i <in_dir> -m <mode> -n <model_name> -c <ckpt_dir>")
      sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print("python3 main.py -i <in_dir> -m <mode> -n <model_name> -c <ckpt_dir>")
      sys.exit()
    elif opt in ("-i", "--in_dir"):
      in_dir = arg
    elif opt in ("-m", "--mode"):
      mode = arg
    elif opt in ("-n", "--model_name"):
      model_name = arg
    elif opt in ("-c", "--ckpt_dir"):
      ckpt_dir = arg


  print("Input directory (dataset): ", in_dir)
  print("Checkpoint directory: ", ckpt_dir)
  print("Mode: ", mode)
  print("Model name: ", model_name)


  # Load Dataset 
  dataset = parse_data(os.path.join(in_dir, mode))
  dataset = Dataset.from_dict(dataset)

  print("------------- Dataset Loaded! --------------")
  print("Len dataset:", len(dataset))
  print("Sample data:", dataset[1])


  #checkpoint paths
  path_model = os.path.join(ckpt_dir, "ckpt_model")
  path_tokenizer = os.path.join(ckpt_dir, "ckpt_tokenizer")


  if mode=='train':

    #Load model
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Fine tune with LORA
    print("------------- Training Started! --------------")

    model = train(model=model,
                  dataset=dataset,
                  tokenizer=tokenizer,
                  out_dir=ckpt_dir)
    
    #save model, tokenizer
    model.save_pretrained(path_model)
    tokenizer.save_pretrained(path_tokenizer)
    print("------------- Model Saved! --------------")

  elif mode=='test':

    #load model and tokenizer from checkpoints
    model = AutoModelForCausalLM.from_pretrained(path_model).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)

    print("------------- Evaluation Started! --------------")
    rouge, bleu = eval_metrics(model, dataset, tokenizer)

    print("------------- Results! --------------")
    print("Rouge: ", rouge)
    print("Bleu", bleu)



if __name__ == '__main__':
    main(sys.argv[1:])