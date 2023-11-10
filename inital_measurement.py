import time
import transformers
from dataclasses import dataclass, field, asdict
from util import datahandler
from typing import Optional, Dict, Sequence
import torch
from datetime import datetime
from transformers import RobertaTokenizer



@dataclass
class DataArguments:
	train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	test_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	vali_file_path: str = field(default='./data/python_splits/val.jsonl', metadata={"help": "path for validation data"})
	report: bool = field(default=False, metadata={"help": "report to wandb or not"})
	save_path: str = field(default='./RRHF_output/datasets/', metadata={"help": "path for saving generated samples"})


@dataclass
class GenerateArguments:
	#train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	#eval_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	batch_size: int = field(default=2, metadata={"help": "batch size"})
	fine_tuning_steps: int = field(default=100, metadata={"help": "fine tuning steps"})
	load_from_model_path: str = field(default='./model_param/model_1_2023-10-18_04-40-16.pth', metadata={"help": "save to model path"})
	db: bool = field(default=False, metadata={"help": "debug mode"})
	num_beams: int = field(default=5, metadata={"help": "number of beam search"})
	num_return_sequences: int = field(default=4, metadata={"help": "num of return sequences for each input ids"})
	output_code_path: str = field(default='./output/', metadata={"help": "path for temperary output code"})
	num_process: int = field(default=20, metadata={"help": "number of process to measure the execution time"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
	output_dir : str = field(default="./RRHF_output/")
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	model_max_length: int = field(
		default=2048,
		metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
	)
	rrhf_weight: float = field(default=100.0)
	length_penalty: float = field(default=1.0)
	only_use_provide: bool = field(default=False)
	only_use_sample: bool = field(default=False)
	report_to: str = field(default='none')
	RL_steps: int = field(default=10)
	per_device_train_batch_size: int = field(default=4)

if __name__ == "__main__":
	start_time = time.time()
	parser = transformers.HfArgumentParser((DataArguments,GenerateArguments,TrainingArguments))
	data_args,generate_args,training_args = parser.parse_args_into_dataclasses()
	parser = transformers.HfArgumentParser((DataArguments,GenerateArguments,TrainingArguments))
	data_args,generate_args,training_args = parser.parse_args_into_dataclasses()
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
	#tokenizer.model_max_length = 300

	if data_args.report:
		wandb.watch(model,log='all')

	
	train_dataset,test_dataset,vali_dataset = datahandler(data_args,generate_args, tokenizer)

	torch.save(train_dataset, data_args.save_path+'train_dataset.pth')
	torch.save(test_dataset, data_args.save_path+'test_dataset.pth')
	torch.save(vali_dataset, data_args.save_path+'vali_dataset.pth')

	end_time = time.time()
	print(end_time-start_time)
