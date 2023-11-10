import time
import transformers
from dataclasses import dataclass, field, asdict
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset, Sampler
from util import remove_special_token
from typing import Optional, Dict, Sequence
import torch
from datetime import datetime
from transformers import RobertaTokenizer,T5ForConditionalGeneration
from tqdm import tqdm
import torch.nn.functional as F
import json


@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
	report: bool = field(default=False, metadata={"help": "report to wandb or not"})
	load_path: str = field(default='./RRHF_output/datasets/', metadata={"help": "path for saving generated samples"})


@dataclass
class GenerateArguments:
	#train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	#eval_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	batch_size: int = field(default=2, metadata={"help": "batch size"})
	load_from_model_path: str = field(default='./model_param/model_1_2023-10-18_04-40-16.pth', metadata={"help": "save to model path"})
	db: bool = field(default=False, metadata={"help": "debug mode"})
	#num_beams: int = field(default=5, metadata={"help": "number of beam search"})
	num_return_sequences: int = field(default=4, metadata={"help": "num of return sequences for each input ids"})
	output_code_path: str = field(default='./output/', metadata={"help": "path for temperary output code"})
	save_result: str = field(default='./generated_list/', metadata={"help": "path for saving generated samples"})
	RL_step : int = field(default=0)


def generate(report, args,tokenizer,model,train_dataloader,num_processes = 2):

	querys = []
	responses = []
	scores = []
	count = 0
	Compile_score = 1
	pass_score = 1.3
	Better_time_score = 2
	number_of_skips = 0
	number_of_better_from_rs = 0
	tra = []
	first_time = time.time()
	for batch in tqdm(train_dataloader,desc="Sampling data"):
		input_ids,input_masks,target_ids,target_masks,problem_ids,v0_times = [t.to(args.device) for t in batch]
		#v0_times = v0_times.tolist()
		greedy_ids = model.generate(
			input_ids, 
			attention_mask=input_masks,
			max_length=tokenizer.model_max_length, 
			do_sample=False, 
		)
		generated_ids = model.generate(input_ids, attention_mask=input_masks, max_length=tokenizer.model_max_length, 
										temperature = 1, top_k = 50,num_beams=1, 
										do_sample=True, num_return_sequences=args.num_return_sequences-2)
		#print(greedy_ids.shape)
		#print(generated_ids.shape)
		#print(target_ids.shape)
		#generated_strs = [tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids]
		if greedy_ids.shape[-1] != tokenizer.model_max_length:
			greedy_ids = F.pad(greedy_ids, (0, tokenizer.model_max_length - greedy_ids.size(-1)),'constant',tokenizer.pad_token_id)
		if generated_ids.shape[-1] != tokenizer.model_max_length:
			generated_ids = F.pad(generated_ids, (0, tokenizer.model_max_length - generated_ids.size(-1)),'constant',tokenizer.pad_token_id)
		#print(generated_ids.shape)
		greedy_ids = greedy_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		generated_ids = generated_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		#print(greedy_ids.shape)
		#print(target_ids.unsqueeze(1).shape)
		generated_ids = torch.cat((generated_ids,greedy_ids),dim=1)
		generated_ids = torch.cat((generated_ids,target_ids.unsqueeze(1)),dim=1)
		#print(generated_ids.shape)
		#print(problem_ids.shape)
		#print(v0_times.shape)
		input_ids = input_ids.tolist()
		generated_ids = generated_ids.tolist()
		problem_ids = problem_ids.tolist()
		v0_times = v0_times.tolist()
		for index, generated_ids4problem in enumerate(generated_ids):
			#print(len(generated_ids4problem))
			problem = 'p'+str(problem_ids[index]).zfill(5)
			v0_time = v0_times[index]
			if v0_time != 1000:
				codes = [remove_special_token(tokenizer.decode(ids[1:], skip_special_tokens=False, 
							clean_up_tokenization_spaces=False),tokenizer) for ids in generated_ids4problem]
				tra.append((input_ids[index],generated_ids4problem,codes,problem,v0_time))
			else:
				number_of_skips += 1
		if count == 2 and args.db:
			break
		count += 1

	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	print('skip rate: ',number_of_skips/(args.batch_size*count))

	with open(args.save_result+'generated_list_{}_{}.json'.format(args.RL_step,current_time), 'w') as f:
		json.dump(tra, f)


if __name__ == "__main__":
	start_time = time.time()
	parser = transformers.HfArgumentParser((DataArguments,GenerateArguments))
	data_args,generate_args = parser.parse_args_into_dataclasses()
	generate_args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_args_dict = asdict(data_args)
	generate_args_dict = asdict(generate_args)
	merged_config = {**data_args_dict, **generate_args_dict}
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	if data_args.report:
		wandb.init(
			# set the wandb project where this run will be logged
			project="RRHF_perf_seperated",
			# track hyperparameters and run metadata
			config=merged_config,
			name=current_time,
		)

	tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	tokenizer.pad_token = tokenizer.eos_token
	model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
	model.load_state_dict(torch.load(generate_args.load_from_model_path))
	model.to(generate_args.device)
	if data_args.report:
		wandb.watch(model,log='all')

	train_dataset = torch.load(data_args.load_path+'train_dataset.pth')
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=generate_args.batch_size,drop_last=True)
	
	output_dataset = generate(data_args.report,generate_args,tokenizer,model,train_dataloader,20)
	end_time = time.time()
	print(end_time-start_time)

