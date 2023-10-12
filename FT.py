import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import datahandler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from itertools import cycle
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup




def fine_tune(args):
	tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
	tokenizer.pad_token = tokenizer.eos_token
	print("Special tokens:", tokenizer.all_special_tokens)
	print("Special token ids:", tokenizer.all_special_ids)
	print(tokenizer.model_max_length)
	train_dataset = datahandler(args, tokenizer)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
	train_dataloader = cycle(train_dataloader)
	model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
	if args.continues:
		model.load_state_dict(torch.load(args.load_from_model_path))
	model.to(args.device)
	optimizer = AdamW(model.parameters(), lr=5e-5)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=args.fine_tuning_steps)
	model.train()

	for step in tqdm(range(args.fine_tuning_steps), total=args.fine_tuning_steps):
		input_ids,input_masks,target_ids,target_masks = [t.to(args.device) for t in next(train_dataloader)]
		output = model(input_ids=input_ids,attention_mask=input_masks,labels=target_ids)
		loss = output.loss
		loss.backward()
		if step % 16 == 0:
			# Update parameters
			optimizer.step()
			optimizer.zero_grad()
			scheduler.step()
			print(loss)

	torch.save(model.state_dict(), args.save_to_model_path)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	## Required parameters  
	parser.add_argument("--train_file_path", default='./data/python_splits/train.jsonl', type=str, help="path for data")  
	parser.add_argument("--batch_size", default=16 , type=int, help="batch size")  
	parser.add_argument("--fine_tuning_steps", default=100 , type=int, help="fine tuning steps")  
	parser.add_argument("--save_to_model_path", default='./model_param/model.pth' , type=str, help="save to model path")  
	parser.add_argument("--db", default=False , type=bool, help="debug mode")  
	parser.add_argument("--continues", default=False , type=bool, help="continue fine tuning")  
	args = parser.parse_args()
	args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(args)
	fine_tune(args)	 
