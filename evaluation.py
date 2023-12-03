from transformers import GenerationConfig
import evaluate


def inference(model, dataset, tokenizer):

  labels = []
  answers = []


  for i, data in enumerate(dataset):

 
    inp, label = data['text'], data['text_label']
    input_ids = tokenizer(inp, return_tensors="pt").input_ids.to('cuda')

    pred_logits = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=20, min_new_tokens=10, pad_token_id=tokenizer.eos_token_id, num_beams=1))
    pred = tokenizer.decode(pred_logits[0], skip_special_tokens=False)

    input_list = inp.split("#")
    pred_list = pred.split("#")
    answer = pred_list[len(input_list)-1]
    answer = pred_list[len(input_list)-1][len(input_list[-1]):]
    
    answers.append(answer)
    labels.append(label)

    if i%100 == 0:
      print("Input:", inp)
      print("Answer:", answer)
      print("Label:",  label)
      print("----------------------------")


  return answers, labels


def eval_metrics(model, dataset, tokenizer):

  answers, labels = inference(model, dataset, tokenizer)

  rouge = evaluate.load('rouge')
  res_rouge = rouge.compute(predictions=answers, references=labels)

  bleu = evaluate.load('bleu')
  res_bleu = bleu.compute(predictions=answers, references=labels)

  return res_rouge, res_bleu


 