from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, Sampler
from dataclasses import dataclass, field, asdict
import transformers
from typing import Optional, Dict, Sequence
from transformers import Trainer,DataCollatorWithPadding
import time
import torch
from transformers.utils import is_datasets_available
import datasets
from transformers.trainer_utils import seed_worker 
import torch.nn.functional as F
from datetime import datetime




@dataclass
class GenerateArguments:
	#train_file_path: str = field(default='./data/python_splits/train.jsonl', metadata={"help": "path for train data"})
	#eval_file_path: str = field(default='./data/python_splits/test.jsonl', metadata={"help": "path for test data"})
	batch_size: int = field(default=2, metadata={"help": "batch size"})
	fine_tuning_steps: int = field(default=100, metadata={"help": "fine tuning steps"})
	load_from_model_path: str = field(default='./model_param/model_4_2023-10-17_08-18-32.pth', metadata={"help": "save to model path"})
	db: bool = field(default=False, metadata={"help": "debug mode"})
	num_beams: int = field(default=5, metadata={"help": "number of beam search"})
	num_return_sequences: int = field(default=4, metadata={"help": "num of return sequences for each input ids"})
	output_code_path: str = field(default='./output/', metadata={"help": "path for temperary output code"})
	load_path: str = field(default='/root/sk/Perf_LLM-main/RRHF_output/datasets/output_dataset.pt', metadata={"help": "path for saving generated samples"})

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
	per_device_train_batch_size: int = field(default=4)
	RL_step: int = field(default=10)


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

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_dataset) -> Dict:
	"""Make dataset and collator for supervised fine-tuning."""
	train_dataset = train_dataset
	data_collator = CustomCollator(tokenizer=tokenizer)
	return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

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
		input_ids = inputs.get('input_ids')
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

if __name__ == "__main__":
	start_time = time.time()
	parser = transformers.HfArgumentParser((GenerateArguments,TrainingArguments))
	generate_args,training_args = parser.parse_args_into_dataclasses()
	generate_args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	generate_args_dict = asdict(generate_args)
	training_args_dict = asdict(training_args)
	merged_config = { **generate_args_dict, **training_args_dict}


	tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	tokenizer.pad_token = tokenizer.eos_token
	model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
	model.load_state_dict(torch.load(generate_args.load_from_model_path))
	model.to(generate_args.device)
	output_dataset = torch.load(generate_args.load_path)
	print(len(output_dataset))
	data_module = make_supervised_data_module(tokenizer=tokenizer, train_dataset=output_dataset)
	trainer = RRHFTrainer(rep = False, model=model, tokenizer=tokenizer, args=training_args, **data_module)
	trainer.train()
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	torch.save(model.state_dict(), './model_param/model_{}_{}.pth'.format(training_args.RL_step,current_time))
	end_time = time.time()
	print(end_time-start_time)
