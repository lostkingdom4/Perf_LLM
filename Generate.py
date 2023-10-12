import torch
import argparse
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from util import datahandler, remove_special_token
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from itertools import cycle
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from compiler.terminal_compiler import TerminalCompiler
from transformers import Trainer

lang2compiler = {
	"python": TerminalCompiler("Python"),
}

def generate(args,tokenizer,model):

	train_dataset = datahandler(args, tokenizer)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
	querys = []
	responses = []
	scores = []
	count = 0

	for batch in tqdm(train_dataloader):
		input_ids,input_masks,target_ids,target_masks = [t.to(args.device) for t in batch]
		generated_ids = model.generate(input_ids, max_length=tokenizer.model_max_length, 
										temperature = 1, top_k = 50,num_beams=1, 
										do_sample=True, num_return_sequences=args.num_return_sequences)
		print(generated_ids.shape)
		for j in range(args.batch_size):
			for _ in range(args.num_return_sequences):
				querys.append(input_ids[j])
		#generated_strs = [tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids]
		generated_ids = generated_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		#print("Special tokens:", tokenizer.all_special_tokens)
		for generated_ids4problem in generated_ids:
			for ids in generated_ids4problem:
				#print(ids)
				responses.append(ids)
			generated_strs = [tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids4problem]
			for generated_str in generated_strs:
				codes = remove_special_token(generated_str,tokenizer)
				#print(type(generated_str))
				#print(remove_special_token(generated_str,tokenizer))
				_,_,did_compile = lang2compiler["python"].compile_code_string(codes)
				print(did_compile)
				if did_compile:
					scores.append(1)
				else:
					scores.append(0)
		if count ==2: break
		count += 1
	querys = torch.stack(querys)
	responses = torch.stack(responses)
	scores = torch.tensor(scores)
	print(querys.shape,responses.shape,scores.shape)
	output_dataset = TensorDataset(querys,responses,scores)

	'''
	TODO: indentation and special_token
	'''
	return output_dataset


if __name__ == "__main__":
	'''
	TODO: combine the parsers
	'''
	parser = argparse.ArgumentParser()
	## Required parameters  
	parser.add_argument("--train_file_path", default='./data/python_splits/train.jsonl', type=str, help="path for data")  
	parser.add_argument("--batch_size", default=2 , type=int, help="batch size")  
	parser.add_argument("--fine_tuning_steps", default=100 , type=int, help="fine tuning steps")  
	parser.add_argument("--load_from_model_path", default='./model_param/model.pth' , type=str, help="save to model path")
	parser.add_argument("--db", default=True , type=bool, help="debug mode")  
	parser.add_argument("--num_beams", default=5 , type=int, help="number of beam search")   
	parser.add_argument("--num_return_sequences", default=3 , type=int, help="num of return sequences for each input ids") 
	parser.add_argument("--output_code_path", default='./output/' , type=str, help="path for temperary output code") 
	args = parser.parse_args()
	args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	tokenizer.pad_token = tokenizer.eos_token
	model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
	model.load_state_dict(torch.load(args.load_from_model_path))
	model.to(args.device)
	output_dataset = generate(args,tokenizer,model)	