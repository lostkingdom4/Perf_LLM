from dataclasses import dataclass, field, asdict
import time
import transformers
from datetime import datetime
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import torch.multiprocessing as multiprocessing
from compiler.terminal_compiler import TerminalCompiler
import torch
from torch.utils.data import TensorDataset



lang2compiler = {
	"python": TerminalCompiler("Python"),
}


@dataclass
class DataArguments:
	load_result: str = field(default='./generated_list/', metadata={"help": "path for saving generated samples"})
	num_process: int = field(default=20, metadata={"help": "number of process to measure the execution time"})
	save_dataset: str = field(default='./RRHF_output/datasets/', metadata={"help": "path for saving output_dataset"})



def process_tra_chunk(tra):
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
	num_pass_test = 0
	number_of_better_from_rs = 0
	#for data_item in tqdm(data, desc='Processing {}'.format(chunk_number)):
	#print(len(data))
	for data_item in data:
		input_id,generated_ids4problem,codes,problem,v0_time = data_item
		#print("len(codes): ", len(codes))
		for index, code in enumerate(codes):
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
		if Better_time_score in scores[-4:-1]:
			number_of_better_from_rs += 1
	return querys, responses, scores, number_of_better_from_rs

def measurement(tra,num_processes = 2):
	querys = []
	responses = []
	scores = []
	count = 0
	Compile_score = 1
	pass_score = 1.3
	Better_time_score = 2
	number_of_skips = 0
	number_of_better_from_rs = 0

	if len(tra)%num_processes == 0:
		length = len(tra)//num_processes
	else:
		length = len(tra)//num_processes + 1

	indice = [(i*length, (i+1)*length) for i in range(num_processes)]
	indice[-1] = ((num_processes-1)*length, len(tra))
	tra_chunk = [(i, tra[index[0]:index[1]]) for i,index in enumerate(indice)]

	if multiprocessing.get_start_method(allow_none=True) is None:
		multiprocessing.set_start_method('spawn')
	with multiprocessing.Pool(num_processes) as pool:
		results = pool.map(process_tra_chunk, tra_chunk)

	for res in results:
		for i in range(len(res[0])):
			querys.append(res[0][i])
			responses.append(res[1][i])
			scores.append(res[2][i])
		number_of_better_from_rs += res[3]

	querys = torch.tensor(querys).cpu()
	responses = torch.tensor(responses).cpu()
	scores = torch.tensor(scores)
	#print(querys.shape[0])
	Compile_rate = torch.sum(scores == Compile_score).item()/scores.shape[0]
	Passing_rate = torch.sum(scores == pass_score).item()/scores.shape[0]
	Optimized_rate = torch.sum(scores == Better_time_score).item()/scores.shape[0]
	print('Compile rate: ',Compile_rate+Passing_rate+Optimized_rate)
	print('Passing rate: ',Passing_rate+Optimized_rate)
	print('Optimized rate: ',Optimized_rate)
	print('number of better from rs: ',number_of_better_from_rs)
	print(querys.shape)
	print(responses.shape)
	print(scores.shape)

	output_dataset = TensorDataset(querys,responses,scores)
	return output_dataset


if __name__ == "__main__":
	start_time = time.time()
	parser = transformers.HfArgumentParser((DataArguments))
	data_args = parser.parse_args_into_dataclasses()
	data_args = data_args[0]
	data_args_dict = asdict(data_args)
	merged_config = {**data_args_dict}
	current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

	tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
	tokenizer.pad_token = tokenizer.eos_token

	with open(data_args.load_result, 'r') as f:
		tra = json.load(f)
	print(len(tra))
	output_dataset = measurement(tra,data_args.num_process) 
	torch.save(output_dataset, data_args.save_dataset+'output_dataset.pt')
	end_time = time.time()
	print(end_time-start_time)

