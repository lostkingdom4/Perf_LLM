import torch
import argparse
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5Config

from util import datahandler, remove_special_token
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset, Sampler
from itertools import cycle
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from compiler.terminal_compiler import TerminalCompiler
from transformers import Trainer

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Sequence
import io
import torch.nn.functional as F
import transformers
import json
import datasets

from transformers import DataCollatorWithPadding
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker 

import wandb
from datetime import datetime
import time

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

RLsteps = int()

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


lang2compiler = {
	"python": TerminalCompiler("Python"),
}

@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
	train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	test_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	vali_file_path: str = field(default='./data/python_splits/val.jsonl', metadata={"help": "path for validation data"})
	report: bool = field(default=False, metadata={"help": "report to wandb or not"})
	save_datasets : bool = field(default=False, metadata={"help": "pre-process and save datasets"})
	load_datasets : bool = field(default=False, metadata={"help": "load datasets"})

@dataclass
class GenerateArguments:
	#train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	#eval_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	batch_size: int = field(default=2, metadata={"help": "batch size"})
	fine_tuning_steps: int = field(default=100, metadata={"help": "fine tuning steps"})
	load_from_model_path: str = field(default='./model_param/model.pth', metadata={"help": "save to model path"})
	db: bool = field(default=False, metadata={"help": "debug mode"})
	num_beams: int = field(default=5, metadata={"help": "number of beam search"})
	num_return_sequences: int = field(default=3, metadata={"help": "num of return sequences for each input ids"})
	output_code_path: str = field(default='./output/', metadata={"help": "path for temperary output code"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
	output_dir : str = field(default="./RRHF_output/")
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	model_max_length: int = field(
		default=512,
		metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
	)
	rrhf_weight: float = field(default=100.0)
	length_penalty: float = field(default=1.0)
	only_use_provide: bool = field(default=False)
	only_use_sample: bool = field(default=False)
	report_to: str = field(default='none')
	RL_steps: int = field(default=10)
	per_device_train_batch_size: int = field(default=4)


def _make_r_io_base(f, mode: str):
	if not isinstance(f, io.IOBase):
		f = open(f, mode=mode)
	return f

def jload(f, mode="r"):
	"""Load a .json file into a dictionary."""
	f = _make_r_io_base(f, mode)
	jdict = json.load(f)
	f.close()
	return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
	"prompt_input": (
		"Below is an instruction that describes a task, paired with an input that provides further context. "
		"Write a response that appropriately completes the request.\n\n"
		"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
	),
	"prompt_no_input": (
		"Below is an instruction that describes a task. "
		"Write a response that appropriately completes the request.\n\n"
		"### Instruction:\n{instruction}\n\n### Response:"
	),
}


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
	"""Collects the state dict and dump to disk."""
	state_dict = trainer.model.state_dict()
	if trainer.args.should_save:
		cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
		del state_dict
		trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
	special_tokens_dict: Dict,
	tokenizer: transformers.PreTrainedTokenizer,
	model: transformers.PreTrainedModel,
):
	"""Resize tokenizer and embedding.

	Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
	"""
	num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if num_new_tokens > 0:
		input_embeddings = model.get_input_embeddings().weight.data
		output_embeddings = model.get_output_embeddings().weight.data

		input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
		output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

		input_embeddings[-num_new_tokens:] = input_embeddings_avg
		output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
	"""Tokenize a list of strings."""
	tokenized_list = [
		tokenizer(
			text,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		)
		for text in strings
	]
	input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
	input_ids_lens = labels_lens = [
		tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
	]
	return dict(
		input_ids=input_ids,
		labels=labels,
		input_ids_lens=input_ids_lens,
		labels_lens=labels_lens,
	)


class ScoreDataset(Dataset):
	"""Dataset for supervised fine-tuning."""

	def __init__(self, data_dict: dict, tokenizer: transformers.PreTrainedTokenizer):
		super(ScoreDataset, self).__init__()
		logging.warning("Loading data...")
		
		# Assuming data_dict is a dictionary where each key points to a list of data points.
		self.data = data_dict

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return dict(input_ids=self.data[i])


def _single_tokenize(text, tokenizer, max_len=None):
	if max_len is None:
		max_len = tokenizer.model_max_length
	toked = tokenizer(
			text,
			return_tensors="pt",
			padding="longest",
			max_length=max_len,
			truncation=True,
		)
	return toked['input_ids'][0]

def stop_response(res):
	stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
	for stop in stops:
		if res.find(stop) >= 0:
			res = res[:res.find(stop)].strip()
	return res

class CustomCollator(DataCollatorWithPadding):
	def __init__(self, tokenizer):
		super(CustomCollator, self).__init__(tokenizer=tokenizer)

	def __call__(self, batch):
		# Separate the input_ids and labels from the batch
		idxs = []
		all_scores = []
		input_ids = []
		score_mask = []
		labels = []
		'''
		decoder_input= self.tokenizer('Faster code: \n',return_tensors="pt")
		decoder_input_ids = decoder_input['input_ids']
		decoder_attention_mask = decoder_input['attention_mask']
		print("decoder_input_ids", decoder_input_ids)
		print("decoder_attention_mask", decoder_attention_mask)
		'''

		for idx, item in enumerate(batch):
			#print(item)
			input_ids.append(item[0])
			labels.append(item[1])
			all_scores.append(item[2])
			idxs.append([idx])
		
		# Pad the sequences
		#padded_input = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt", padding="max_length", max_length=512)
		
		# Convert labels to tensor
		#print("labels",labels)
		input_ids = torch.stack(input_ids)
		labels = torch.stack(labels)
		#print("input_ids.shape", input_ids.shape)
		#decoder_input_ids = torch.unsqueeze()
		#decoder_input_ids = decoder_input_ids.repeat(input_ids.shape[0], 1)
		#decoder_attention_mask = decoder_attention_mask.repeat(input_ids.shape[0], 1)
		#print("decoder_attention_mask", decoder_attention_mask.shape)
		return dict(
			input_ids=input_ids,
			attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
			labels=labels,
			idxs=torch.LongTensor(idxs),
			scores=torch.FloatTensor(all_scores),
			#decoder_input_ids = decoder_input_ids,
			#decoder_attention_mask = decoder_attention_mask
		)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_dataset) -> Dict:
	"""Make dataset and collator for supervised fine-tuning."""
	train_dataset = train_dataset
	data_collator = CustomCollator(tokenizer=tokenizer)
	return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class GroupedRandomSampler(Sampler):
	def __init__(self, dataset, group_size):
		self.dataset = dataset
		self.group_size = group_size
		self.num_groups = len(self.dataset) // self.group_size
		if len(self.dataset) % self.group_size != 0:
			raise ValueError("Trainer: train_dataset cannot divided by the group_size")

	def __iter__(self):
		group_indices = torch.randperm(self.num_groups).tolist()
		return iter([group * self.group_size + i for group in group_indices for i in range(self.group_size)])

	def __len__(self):
		return len(self.dataset)

class RRHFTrainer(Trainer):
	
	def __init__(self, rep = False, *args, **kwargs):
		super().__init__(*args, **kwargs)  # Call the init of the parent class
		self.rep = rep 

	def get_train_dataloader(self):
		train_dataset = self.train_dataset
		data_collator = self.data_collator
		if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
			train_dataset = self._remove_unused_columns(train_dataset, description="training")
		else:
			data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
		dataloader_params = {
			"batch_size": self._train_batch_size,
			"collate_fn": data_collator,
			"num_workers": self.args.dataloader_num_workers,
			"pin_memory": self.args.dataloader_pin_memory,
		}
		if not isinstance(train_dataset, torch.utils.data.IterableDataset):
			dataloader_params["sampler"] = GroupedRandomSampler(self.train_dataset, self._train_batch_size)
			dataloader_params["drop_last"] = self.args.dataloader_drop_last
			dataloader_params["worker_init_fn"] = seed_worker
		return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

	def gather_logits_labels(self, logits, labels):

		mask = (labels != -100).float()
		#print(mask)
		new_logits = logits.clone()  # Create a copy to avoid in-place modification
		labels[labels == -100] = 0 
		output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
		output = output * mask # B * L
		return output

	def get_score(self, logit_label, labels):
		mask = (labels != -100).float()
		length = mask.sum(-1)
		scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
		return scores

	def rrhf_loss(self, scores, idxs, rw_scores):
		#print(rw_scores)
		#print(scores.unsqueeze(0))
		#print(scores.unsqueeze(-1))
		diff = scores.unsqueeze(0) - scores.unsqueeze(-1) # b * b
		rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1) # b * b
		# select terms that have better rw performance but less probability
		aval = torch.bitwise_and(rw_diff > 0, diff < 0)
		'''
		print(diff)
		print(rw_diff)
		print(aval)
		print(-diff[aval].sum())
		'''
		return -diff[aval].sum()

	def sft_loss(self, logit_label, idxs, rw_scores):
		max_idx = torch.argmax(rw_scores)
		return -logit_label[max_idx].mean()

	def compute_loss(self, model, inputs, return_outputs=False):
		#print(inputs.keys())
		#print(inputs.get('input_ids').shape)
		#output_ids = model.generate(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),max_length=tokenizer.model_max_length)
		output = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),
						labels=inputs.get('labels')) # (batch * cand) * L * V		
		#logits = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),decoder_input_ids=inputs.get('decoder_input_ids'),decoder_attention_mask=inputs.get('decoder_attention_mask'))[0] # (batch * cand) * L * V
		#print(output.logits)
		#print(output.logits.shape)
		logits = output.logits
		logits = F.log_softmax(logits, dim=-1)
		#print(logits.shape)
		logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
		#print(logit_label)
		scores = self.get_score(logit_label, inputs.get("labels"))
		#print(scores)
		rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
		sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
		#print(self.args.rrhf_weight * rrhf_loss,sft_loss)
		loss = self.args.rrhf_weight * rrhf_loss + sft_loss
		#exit()
		#print(loss, scores)
		if self.rep:
			wandb.log({
				'fine_tune_loss': loss,
				'fine_tune_rrhf_loss': rrhf_loss,
				'fine_tune_sft_loss': sft_loss,
				'rl_step': RLsteps  # Logging the RL step can be helpful
			})
		return (loss, scores) if return_outputs else loss

def generate(report, args,tokenizer,model,train_dataloader):

	querys = []
	responses = []
	scores = []
	count = 0
	Compile_score = 1
	pass_score = 1.3
	Better_time_score = 2

	for batch in tqdm(train_dataloader,desc="Sampling data"):
		input_ids,input_masks,target_ids,target_masks,problem_ids,v0_times = [t.to(args.device) for t in batch]
		generated_ids = model.generate(input_ids, max_length=tokenizer.model_max_length, 
										temperature = 1, top_k = 50,num_beams=1, 
										do_sample=True, num_return_sequences=args.num_return_sequences)
		#print(generated_ids.shape)
		#print(target_ids.shape)
		for j in range(args.batch_size):
			for _ in range(args.num_return_sequences+1):
				querys.append(input_ids[j])
		#generated_strs = [tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in generated_ids]
		if generated_ids.shape[-1] != tokenizer.model_max_length:
			generated_ids = F.pad(generated_ids, (0, 512 - generated_ids.size(-1)),'constant',tokenizer.pad_token_id)
		#print(generated_ids.shape)
		generated_ids = generated_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		generated_ids = torch.cat((generated_ids,target_ids.unsqueeze(1)),dim=1)
		#print(generated_ids.shape)
		for index, generated_ids4problem in enumerate(generated_ids):
			problem = 'p'+str(problem_ids[index].item()).zfill(5)
			v0_time = v0_times[index]
			for ids in generated_ids4problem:
				#print(ids)
				responses.append(ids)
			generated_strs = [tokenizer.decode(ids[1:], skip_special_tokens=False, 
						clean_up_tokenization_spaces=False) for ids in generated_ids4problem]
			for generated_str in generated_strs:
				codes = remove_special_token(generated_str,tokenizer)
				#print(codes)
				#print(type(generated_str))
				#print(remove_special_token(generated_str,tokenizer))
				a,b,did_compile = lang2compiler["python"].compile_code_string(codes,problem)
				#print(a)
				#print(b)
				#print(did_compile)
				if did_compile:
					a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(codes,problem)
					if pass_test:
						if v0_time > elapsed_time:
							scores.append(Better_time_score)
						else:
							scores.append(pass_score)
					else:
						scores.append(Compile_score)
				else:
					scores.append(0)
		if count == 2 and args.db:
			break
		elif count ==20:
			break

		count += 1
	querys = torch.stack(querys).cpu()
	responses = torch.stack(responses).cpu()
	scores = torch.tensor(scores)
	#print(querys.shape[0])
	Compile_rate = torch.sum(scores == Compile_score).item()/scores.shape[0]
	Passing_rate = torch.sum(scores == pass_score).item()/scores.shape[0]
	Optimized_rate = torch.sum(scores == Better_time_score).item()/scores.shape[0]
	print('Compile rate: ',Compile_rate+Passing_rate+Optimized_rate)
	print('Passing rate: ',Passing_rate+Optimized_rate)
	print('Optimized rate: ',Optimized_rate)
	#print(querys.shape)
	#print(responses.shape)
	if report:
		wandb.log({
			'Compile rate': Compile_rate+Passing_rate+Optimized_rate,
			'Passing rate': Passing_rate+Optimized_rate,
			'Optimized rate: ': Optimized_rate,
			'rl_step': RLsteps  # Logging the RL step can be helpful
		})

	output_dataset = TensorDataset(querys,responses,scores)

	'''
	TODO: indentation and special_token
	'''
	return output_dataset

def Testorvali(args,tokenizer,model,dataloader,description):
	compile_scores = []
	pass_test_scores = []
	Optimized_scores = []
	count = 0
	for batch in tqdm(dataloader,desc=description):
		input_ids,input_masks,target_ids,target_masks,problem_ids, v0_times = [t.to(args.device) for t in batch]
		generated_ids = model.generate(input_ids, max_length=tokenizer.model_max_length, 
										temperature = 1, top_k = 50,num_beams=1, 
										do_sample=True, num_return_sequences=args.num_return_sequences+1)
		if generated_ids.shape[-1] != tokenizer.model_max_length:
			generated_ids = F.pad(generated_ids, (0, 512 - generated_ids.size(-1)),'constant',tokenizer.pad_token_id)
		generated_ids = generated_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		for index,generated_ids4problem in enumerate(generated_ids):
			problem = 'p'+str(problem_ids[index].item()).zfill(5)
			v0_time = v0_times[index]
			generated_strs = [tokenizer.decode(ids[1:], skip_special_tokens=False, 
				clean_up_tokenization_spaces=False) for ids in generated_ids4problem]
			onecompile = False
			onerun = False
			oneoptimized = False
			for generated_str in generated_strs:
				codes = remove_special_token(generated_str,tokenizer)
				#print(codes)
				#print(type(generated_str))
				#print(remove_special_token(generated_str,tokenizer))
				a,b,did_compile = lang2compiler["python"].compile_code_string(codes,problem)
				#print(a)
				#print(b)
				#print(did_compile)
				if did_compile:
					onecompile = True
					a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(codes,problem)
					if pass_test:
						onerun = True
						if v0_time > elapsed_time:
							oneoptimized = True
			
			compile_scores.append(onecompile)
			pass_test_scores.append(onerun)
			Optimized_scores.append(oneoptimized)
		if description == "Validating" and count == 50: break
		#if description == "Testing" and count == 19: break
		count += 1
	#print(scores)
	compile_rate = sum(compile_scores) / len(compile_scores)
	pass_rate = sum(pass_test_scores) / len(pass_test_scores)
	Optimized_rate = sum(Optimized_scores) / len(Optimized_scores)
	print(compile_rate,pass_rate,Optimized_rate)
	return compile_rate,pass_rate,Optimized_rate

def test(output_dataset):

	# Constants
	IGNORE_INDEX = -100
	'''
	# Sample instance
	instances = [
		{
			'input_ids': {
				'query': 'What is your name?',
				'responses': ['I am a bot.', 'Name is ChatGPT.', 'Call me Assistant.'],
				'scores': [0.8, 0.7, 0.9]
			}
		},
		{
			'input_ids': {
				'query': 'Where are you from?',
				'responses': ['I am from the cloud.', 'Virtual world is my home.'],
				'scores': [0.85, 0.75]
			}
		}
	]
	'''

	# Initializing the DataCollator
	collator = CustomCollator(tokenizer=tokenizer)

	# Testing
	collated_data = collator(output_dataset)
	print(collated_data)
	



if __name__ == "__main__":
	start_time = time.time()
	'''
	TODO: combine the parsers
	'''
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
	'''
	parser = transformers.HfArgumentParser((DataArguments,GenerateArguments,TrainingArguments))
	data_args,generate_args,training_args = parser.parse_args_into_dataclasses()
	#training_args.per_device_train_batch_size = 4
	#if generate_args.db: training_args.RL_steps = 1
	generate_args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_args_dict = asdict(data_args)
	generate_args_dict = asdict(generate_args)
	training_args_dict = asdict(training_args)
	merged_config = {**data_args_dict, **generate_args_dict, **training_args_dict}

	if data_args.report:
		wandb.init(
			# set the wandb project where this run will be logged
			project="RRHF_perf",
			# track hyperparameters and run metadata
			config=merged_config,
			name=current_time,
		)

	tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	tokenizer.pad_token = tokenizer.eos_token
	model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
	model.load_state_dict(torch.load(generate_args.load_from_model_path))
	model.to(generate_args.device)
	#print("Special tokens:", tokenizer.all_special_tokens)
	#print(tokenizer.special_tokens_map)
	#print("Special token ids:", tokenizer.all_special_ids)
	#print(tokenizer.model_max_length)
	if data_args.report:
		wandb.watch(model,log='all')

	if data_args.load_datasets:
		train_dataset = torch.load(training_args.output_dir+'datasets/train_dataset.pth')
		test_dataset = torch.load(training_args.output_dir+'datasets/test_dataset.pth')
		vali_dataset = torch.load(training_args.output_dir+'datasets/vali_dataset.pth')
	else:
		train_dataset,test_dataset,vali_dataset = datahandler(data_args,generate_args, tokenizer)

	if data_args.save_datasets:
		torch.save(train_dataset, training_args.output_dir+'datasets/train_dataset.pth')
		torch.save(test_dataset, training_args.output_dir+'datasets/test_dataset.pth')
		torch.save(vali_dataset, training_args.output_dir+'datasets/vali_dataset.pth')

	train_sampler = RandomSampler(train_dataset)
	test_sampler = SequentialSampler(test_dataset)
	vali_sampler = RandomSampler(vali_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=generate_args.batch_size,drop_last=True)
	test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=generate_args.batch_size,drop_last=True)
	vali_dataloader = DataLoader(vali_dataset, sampler=vali_sampler, batch_size=generate_args.batch_size,drop_last=True)

	
	compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,test_dataloader,"Testing")
	if data_args.report:
		wandb.log({
			'Testing Compile rate': compile_rate,
			'Testing Pass rate': pass_rate,
			'Testing Optimized rate': Optimized_rate,
			'Testing rl_step': RLsteps  # Logging the RL step can be helpful
		})
	

	for rsteps in range(training_args.RL_steps):
		RLsteps = rsteps
		output_dataset = generate(data_args.report,generate_args,tokenizer,model,train_dataloader)
		#print(len(output_dataset))	
		#test(output_dataset)
		data_module = make_supervised_data_module(tokenizer=tokenizer, train_dataset=output_dataset)
		#print(data_module)
		trainer = RRHFTrainer(rep = data_args.report, model=model, tokenizer=tokenizer, args=training_args, **data_module)
		trainer.train()
		#trainer.save_state()
		if data_args.report:
			wandb.log({
						**trainer.state.log_history[0],
						'rl_step': RLsteps  # Logging the RL step can be helpful
			})
		safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
		if rsteps%4==0:
			compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,vali_dataloader,"Validating")
			if data_args.report:
				wandb.log({
					'Validating Compile rate': compile_rate,
					'Validating Pass rate': pass_rate,
					'Validating Optimized rate': Optimized_rate,
					'Validating rl_step': RLsteps  # Logging the RL step can be helpful
				})
	compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,test_dataloader,"Testing")
	if data_args.report:
		wandb.log({
			'Testing Compile rate': compile_rate,
			'Testing Pass rate': pass_rate,
			'Testing Optimized rate': Optimized_rate,
			'Testing rl_step': RLsteps  # Logging the RL step can be helpful
		})
	end_time = time.time()
	print(end_time-start_time)






