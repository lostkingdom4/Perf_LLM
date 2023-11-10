 

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
	tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono",padding_side='left')
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.model_max_length = 1024
	model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
	model.load_state_dict(torch.load(generate_args.load_from_model_path))
	model.to(generate_args.device)

	data_module = make_supervised_data_module(tokenizer=tokenizer, train_dataset=output_dataset)
	trainer = RRHFTrainer(rep = data_args.report, model=model, tokenizer=tokenizer, args=training_args, **data_module)
		trainer.train()