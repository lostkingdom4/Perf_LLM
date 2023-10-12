import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from compiler.terminal_compiler import TerminalCompiler

lang2compiler = {
    "python": TerminalCompiler("Python"),
}


def read_data(file_path):

    # Define a list to hold the parsed JSON objects
    data_list = []
    # Open the .jsonl file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON data for each line and append to the list
            data_list.append(json.loads(line))

    # Now, data_list contains all the JSON objects from the .jsonl file
    #print(data_list[0])
    
    problem_ids = dict()
    languages = []
    problem_char = 0
    for data in data_list:
        if data['problem_id'] not in problem_ids:
            problem_ids[data['problem_id']] = 0
        else:
            pass
            problem_ids[data['problem_id']] += 1
        if data['code_v0_num_chars'] > problem_char or data['code_v1_num_chars'] > problem_char:
            problem_char = max(data['code_v0_num_chars'],data['code_v1_num_chars'])
    print(problem_char)
    return data_list

def generate_dataset_from_list(data_list,db,tokenizer,description):
    code_v0s = []
    code_v1s = []
    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    program_ids = []
    times = []
    test = "Processing " + description + " data"
    if db:
        i = 0
    for data in tqdm(data_list, desc=test):
        code_v0s.append(data['code_v0_no_empty_lines'])
        code_v1s.append(data['code_v1_no_empty_lines'])
        codes = data['code_v0_no_empty_lines']
        #print(codes)
        a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(codes,data['problem_id'])
        #print(elapsed_time)
        if pass_test:
            times.append(elapsed_time)
        else:
            times.append(100)
        input_data = 'Write faster version:\n'+data['code_v0_no_empty_lines']
        input_tokenized = tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True)
        target_tokenized = tokenizer(data['code_v1_no_empty_lines'], return_tensors="pt", padding="max_length", truncation=True)
        input_ids.append(input_tokenized["input_ids"][-1])
        input_masks.append(input_tokenized["attention_mask"][-1])
        target_ids.append(target_tokenized["input_ids"][-1])
        target_masks.append(target_tokenized["attention_mask"][-1])
        program_ids.append(int(data['problem_id'][1:]))
        #if db and i == 32: break
        #if db: i += 1
    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    target_ids = torch.stack(target_ids)
    target_masks = torch.stack(target_masks)
    program_ids = torch.tensor(program_ids)
    times = torch.tensor(times)
    #print(input_ids.shape)
    #print(input_masks.shape)
    #print(target_ids.shape)
    #print(target_masks.shape)
    output_dataset = TensorDataset(input_ids,input_masks,target_ids,target_masks,program_ids,times)
    print(len(output_dataset))
    return output_dataset


def datahandler(data_args, generate_args, tokenizer):
    train_data_list = read_data(data_args.train_file_path)
    test_data_list = read_data(data_args.test_file_path)
    vali_data_list = read_data(data_args.vali_file_path)
    train_dataset = generate_dataset_from_list(train_data_list,generate_args.db,tokenizer,"Train")
    test_dataset = generate_dataset_from_list(test_data_list,generate_args.db,tokenizer,"test")
    vali_dataset = generate_dataset_from_list(vali_data_list,generate_args.db,tokenizer,"vali")

    return train_dataset,test_dataset,vali_dataset

def remove_special_token(generated_str,tokenizer):
    special_tokens = [tokenizer.cls_token, tokenizer.pad_token, tokenizer.eos_token, '<pad>']  # Add other special tokens if needed
    #print(special_tokens)
    #print(generated_str[0])
    #print(special_tokens)
    decoded_text = generated_str
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "")
    return decoded_text
