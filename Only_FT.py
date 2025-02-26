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

import torch.multiprocessing as multiprocessing
from functools import partial

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
	load_from_model_path: str = field(default='./model_param/model_4_2023-10-18_04-50-19.pth', metadata={"help": "save to model path"})
	db: bool = field(default=False, metadata={"help": "debug mode"})
	num_beams: int = field(default=5, metadata={"help": "number of beam search"})
	num_return_sequences: int = field(default=4, metadata={"help": "num of return sequences for each input ids"})
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
			dataloader_params["sampler"] = SequentialSampler(self.train_dataset)
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
		#max_idx = torch.argmax(rw_scores)
		return -logit_label.mean()

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
		#scores = self.get_score(logit_label, inputs.get("labels"))
		#print(scores)
		#rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
		sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
		#print(self.args.rrhf_weight * rrhf_loss,sft_loss)
		loss = sft_loss
		#print(loss.shape)
		#exit()
		#print(loss, scores)
		if self.rep:
			wandb.log({
				'fine_tune_loss': loss,
				#'fine_tune_rrhf_loss': rrhf_loss,
				'fine_tune_sft_loss': sft_loss,
				'rl_step': RLsteps  # Logging the RL step can be helpful
			})
		return (loss, scores) if return_outputs else loss

def execute_chunk(args,tokenizer,problem,v0_time):
	number_of_better_from_rs = 0
	chunk_number, codes = args
	scores = []
	Compile_score = 1
	pass_score = 1.3
	Better_time_score = 2
	for code in codes:
		_,_,did_compile = lang2compiler["python"].compile_code_string(code,problem,chunk_number)
		if did_compile:
			a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(code,problem,chunk_number)
			if pass_test:
				if v0_time > elapsed_time:
					scores.append(Better_time_score)
				else:
					scores.append(pass_score)
			else:
				scores.append(Compile_score)
		else:
			scores.append(0)
	if Better_time_score in scores[-4:-1]:
		number_of_better_from_rs_for_this_input += 1
	return scores,number_of_better_from_rs

def execute_in_multi_processes(codes:list, problem:str, v0_time, tokenizer, num_processes=4):
	#print(codes)
	scores = []
	number_of_better_from_rs=0
	if len(codes)%num_processes == 0:
		length = len(codes)//num_processes
	else:
		length = len(codes)//num_processes + 1

	indice = [(i*length, (i+1)*length) for i in range(num_processes)]
	indice[-1] = ((num_processes-1)*length, len(codes))
	#print(indice)
	codes_chunks = [(i, codes[index[0]:index[1]]) for i,index in enumerate(indice)]

	if multiprocessing.get_start_method(allow_none=True) is None:
		multiprocessing.set_start_method('spawn')
	with multiprocessing.Pool(num_processes) as pool:
		partial_func = partial(execute_chunk, tokenizer=tokenizer,problem=problem,v0_time=v0_time)
		results = pool.map(partial_func, codes_chunks)
	for res in results:
		for i in range(len(res[0])):
			scores.append(res[0][i])
		number_of_better_from_rs += res[1]
	return scores,number_of_better_from_rs

def process_tra_chunk(tra,tokenizer):
	# This function will handle the tokenization and processing of a chunk of data
	#tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	#tokenizer.pad_token = tokenizer.eos_token
	chunk_number, data = tra
	#print("Start {}".format(chunk_number))
	#code_v0s = []
	#code_v1s = []
	Compile_score = 1
	pass_score = 1.3
	Better_time_score = 2
	number_of_skips = 0
	querys = []
	responses = []
	program_ids = []
	scores = []
	g_scores = []
	num_pass_test = 0
	number_of_better_from_rs = 0
	#for data_item in tqdm(data, desc='Processing {}'.format(chunk_number)):
	#print(len(data))
	for data_item in data:
		input_id,generated_ids4problem,codes,problem,v0_time = data_item
		#print("len(codes): ", len(codes))
		for index, code in enumerate(codes):
			if index == 0:
				a,b,did_compile = lang2compiler["python"].compile_code_string(code,problem,chunk_number) 
				if did_compile:
					a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(code,problem,chunk_number)
					if pass_test:
						if v0_time > elapsed_time:
							g_scores.append(Better_time_score)
						else:
							g_scores.append(pass_score)
					else:
						g_scores.append(Compile_score)  
				else:
					g_scores.append(0) 
			else:
				querys.append(input_id)
				responses.append(generated_ids4problem[index])
				a,b,did_compile = lang2compiler["python"].compile_code_string(code,problem,chunk_number) 
				if did_compile:
					a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(code,problem,chunk_number)
					if pass_test:
						if v0_time > elapsed_time:
							scores.append(Better_time_score)
						else:
							scores.append(pass_score)
					else:
						scores.append(Compile_score)  
				else:
					scores.append(0)     
	#print("len(querys): ", len(querys))
	#print("len(responses): ",len(responses))
	#print("len(scores): ",len(scores))
	#print("number_of_better_from_rs: ", number_of_better_from_rs)
	return querys, responses, scores, number_of_better_from_rs, g_scores


def generate(report, args,tokenizer,model,train_dataloader,num_processes = 2):

	querys = []
	responses = []
	scores = []
	g_scores = []
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
		greedy_ids = model.generate(
			input_ids, 
			max_length=tokenizer.model_max_length, 
			do_sample=False, 
		)
		if greedy_ids.shape[-1] != tokenizer.model_max_length:
			greedy_ids = F.pad(greedy_ids, (0, 512 - greedy_ids.size(-1)),'constant',tokenizer.pad_token_id)
		greedy_ids = greedy_ids.view(args.batch_size,-1,tokenizer.model_max_length)
		generated_ids = torch.cat((greedy_ids,target_ids.unsqueeze(1)),dim=1)
		input_ids = input_ids.tolist()
		problem_ids = problem_ids.tolist()
		generated_ids = generated_ids.tolist()
		v0_times = v0_times.tolist()
		for index, generated_ids4problem in enumerate(generated_ids):
			#print("len(generated_ids4problem) ",len(generated_ids4problem))
			problem = 'p'+str(problem_ids[index]).zfill(5)
			v0_time = v0_times[index]
			if v0_time != 1000:
				codes = [remove_special_token(tokenizer.decode(ids[1:], skip_special_tokens=False, 
							clean_up_tokenization_spaces=False),tokenizer) for ids in generated_ids4problem]
				#print(len(codes))
				tra.append((input_ids[index],generated_ids4problem,codes,problem,v0_time))
			else:
				number_of_skips += 1
		if count == 2 and args.db:
			break
		count += 1

	#print(len(tra))
	if len(tra)%num_processes == 0:
		length = len(tra)//num_processes
	else:
		length = len(tra)//num_processes + 1

	#print(length)
	indice = [(i*length, (i+1)*length) for i in range(num_processes)]
	#print(indice)
	indice[-1] = ((num_processes-1)*length, len(tra))
	tra_chunk = [(i, tra[index[0]:index[1]]) for i,index in enumerate(indice)]
	second_time = time.time()

	if multiprocessing.get_start_method(allow_none=True) is None:
		multiprocessing.set_start_method('spawn')
	with multiprocessing.Pool(num_processes) as pool:
		partial_func = partial(process_tra_chunk, tokenizer=tokenizer)
		results = pool.map(partial_func, tra_chunk)
	
	for res in results:
		for i in range(len(res[0])):
			querys.append(res[0][i])
			responses.append(res[1][i])
			scores.append(res[2][i])
			g_scores.append(res[4][i])
		number_of_better_from_rs += res[3]
	'''
				for ids in generated_ids4problem:
					#print(ids)
					querys.append(input_ids[index])
					responses.append(ids)
					generated_strs.append(tokenizer.decode(ids[1:], skip_special_tokens=False, 
							clean_up_tokenization_spaces=False))
					#codes.append(remove_special_token(generated_strs[-1],tokenizer))
				#scores,number_of_better_from_rs = execute_in_multi_processes(codes,problem,v0_time,tokenizer,args.num_return_sequences+1)
				#print(scores,number_of_better_from_rs)
			
			###	for ids in generated_ids4problem:
					#print(ids)
					querys.append(input_ids[index])
					responses.append(ids)
				generated_strs = [tokenizer.decode(ids[1:], skip_special_tokens=False, 
							clean_up_tokenization_spaces=False) for ids in generated_ids4problem]
				

				
				for index_gen, generated_str in enumerate(generated_strs):
					code = remove_special_token(generated_str,tokenizer)
					codes.append(code)
					#print(codes)
					#print(type(generated_str))
					#print(remove_special_token(generated_str,tokenizer))
					a,b,did_compile = lang2compiler["python"].compile_code_string(code,problem,index_gen)
					#print(a)
					#print(b)
					#print(did_compile)
					if did_compile:
						a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(code,problem,index_gen)
						if pass_test:
							if v0_time > elapsed_time:
								scores.append(Better_time_score)
							else:
								scores.append(pass_score)
						else:
							scores.append(Compile_score)
					else:
						scores.append(0)
				if Better_time_score in scores[-4:-1]:
					number_of_better_from_rs += 1
			else:
				number_of_skips += 1
		if count == 2 and args.db:
			break

		#count += 1
		#third_time = time.time()
		#print('time1: ', second_time-first_time)
		#print('time2: ', third_time- second_time)
	'''
	third_time = time.time()
	print('time1: ', second_time-first_time)
	print('time2: ', third_time- second_time)
	print('skip rate: ',number_of_skips/(args.batch_size*count))
	querys = torch.tensor(querys).cpu()
	responses = torch.tensor(responses).cpu()
	scores = torch.tensor(scores)
	g_scores = torch.tensor(g_scores)
	#print(querys.shape[0])
	Compile_rate = torch.sum(g_scores == Compile_score).item()/g_scores.shape[0]
	Passing_rate = torch.sum(g_scores == pass_score).item()/g_scores.shape[0]
	Optimized_rate = torch.sum(g_scores == Better_time_score).item()/g_scores.shape[0]
	print('Compile rate: ',Compile_rate+Passing_rate+Optimized_rate)
	print('Passing rate: ',Passing_rate+Optimized_rate)
	print('Optimized rate: ',Optimized_rate)
	print('rate of better from rs: ',number_of_better_from_rs/(args.batch_size*count))
	print(querys.shape)
	print(responses.shape)
	print(scores.shape)
	print(g_scores.shape)
	if report:
		wandb.log({
			'Compile rate': Compile_rate+Passing_rate+Optimized_rate,
			'Passing rate': Passing_rate+Optimized_rate,
			'Optimized rate: ': Optimized_rate,
			'rate of better from rs':number_of_better_from_rs/(args.batch_size*count),
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
	#count = 0
	for batch in tqdm(dataloader,desc=description):
		input_ids,input_masks,target_ids,target_masks,problem_ids, v0_times = [t.to(args.device) for t in batch]
		generated_ids = model.generate(
			input_ids, 
			max_length=tokenizer.model_max_length, 
			do_sample=False, 
			num_beams=4,
			num_return_sequences=2
		)
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
			for index_gen, generated_str in enumerate(generated_strs):
				codes = remove_special_token(generated_str,tokenizer)
				#print(codes)
				#print(type(generated_str))
				#print(remove_special_token(generated_str,tokenizer))
				a,b,did_compile = lang2compiler["python"].compile_code_string(codes,problem,index_gen)
				#print(a)
				#print(b)
				#print(did_compile)
				if did_compile:
					onecompile = True
					a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(codes,problem,index_gen)
					if pass_test:
						onerun = True
						if v0_time > elapsed_time:
							oneoptimized = True
			
			compile_scores.append(onecompile)
			pass_test_scores.append(onerun)
			Optimized_scores.append(oneoptimized)
		#if description == "Validating" and count == 50: break
		#if description == "Testing" and count == 19: break
		#count += 1
	#print(scores)
	compile_rate = sum(compile_scores) / len(compile_scores)
	pass_rate = sum(pass_test_scores) / len(pass_test_scores)
	Optimized_rate = sum(Optimized_scores) / len(Optimized_scores)
	print(compile_rate,pass_rate,Optimized_rate)
	return compile_rate,pass_rate,Optimized_rate


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
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
		#test_dataset,vali_dataset = datahandler(data_args,generate_args, tokenizer)
		train_dataset,test_dataset,vali_dataset = datahandler(data_args,generate_args, tokenizer)

	if data_args.save_datasets:
		torch.save(train_dataset, training_args.output_dir+'datasets/train_dataset.pth')
		torch.save(test_dataset, training_args.output_dir+'datasets/test_dataset.pth')
		torch.save(vali_dataset, training_args.output_dir+'datasets/vali_dataset.pth')

	
	train_sampler = RandomSampler(train_dataset)
	test_sampler = SequentialSampler(test_dataset)
	vali_sampler = SequentialSampler(vali_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=generate_args.batch_size,drop_last=True)
	test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=generate_args.batch_size,drop_last=True)
	vali_dataloader = DataLoader(vali_dataset, sampler=vali_sampler, batch_size=generate_args.batch_size,drop_last=True)

	'''
	compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,test_dataloader,"Testing")
	#compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,vali_dataloader,"Validating")
	
	if data_args.report:
		wandb.log({
			'Testing Compile rate': compile_rate,
			'Testing Pass rate': pass_rate,
			'Testing Optimized rate': Optimized_rate,
			'rl_step': RLsteps  # Logging the RL step can be helpful
		})
	'''
	onetime = False
	for rsteps in range(training_args.RL_steps):
		RLsteps = rsteps
		if onetime:
			output_dataset = torch.load('./dataset_train/saved_dataset_1_2023-10-17_03-32-43.pt')
			onetime = False
		else:
			output_dataset = generate(data_args.report,generate_args,tokenizer,model,train_dataloader,32)
		onetime = False
		current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		#torch.save(output_dataset, './dataset_train/latest_data.pt')
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
		if (rsteps+1)%5==0:
			current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			torch.save(model.state_dict(), './model_param/model_{}_{}.pth'.format(rsteps,current_time))
			compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,vali_dataloader,"Validating")
			if data_args.report:
				wandb.log({
					'Validating Compile rate': compile_rate,
					'Validating Pass rate': pass_rate,
					'Validating Optimized rate': Optimized_rate,
					'rl_step': RLsteps  # Logging the RL step can be helpful
				})
	
	
	compile_rate,pass_rate,Optimized_rate = Testorvali(generate_args,tokenizer,model,test_dataloader,"Testing")
	if data_args.report:
		wandb.log({
			'Testing Compile rate': compile_rate,
			'Testing Pass rate': pass_rate,
			'Testing Optimized rate': Optimized_rate,
			'rl_step': RLsteps  # Logging the RL step can be helpful
		})
	
	end_time = time.time()
	print(end_time-start_time)
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	torch.save(model.state_dict(), './model_param/model_{}_{}.pth'.format(training_args.RL_steps,current_time))
	
