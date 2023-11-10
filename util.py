import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from compiler.terminal_compiler import TerminalCompiler
import torch.multiprocessing as multiprocessing
from functools import partial
from transformers import RobertaTokenizer
import time

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
    num_pass_test = 0
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
            num_pass_test += 1
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
    print(num_pass_test)
    return output_dataset

def process_data_chunk(args,tokenizer):
    # This function will handle the tokenization and processing of a chunk of data
    #tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    #tokenizer.pad_token = tokenizer.eos_token
    chunk_number, data = args
    #print("Start {}".format(chunk_number))
    #print(data[0])
    #code_v0s = []
    #code_v1s = []
    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    program_ids = []
    times = []
    num_pass_test = 0
    #for data_item in tqdm(data, desc='Processing {}'.format(chunk_number)):
    for data_item in data:
        #code_v0s.append(data_item['code_v0_no_empty_lines'])
        #code_v1s.append(data_item['code_v1_no_empty_lines'])
        codes = data_item['code_v0_no_empty_lines']
        #print("generating")
        a,b,pass_test,elapsed_time = lang2compiler["python"].execute_code_string(codes,data_item['problem_id'],chunk_number)
        #print(elapsed_time)
        #pass_test = True
        #elapsed_time = 1
        if pass_test:
            times.append(elapsed_time)
            num_pass_test += 1
        else:
            times.append(1000)
        input_data = 'Write faster version:\n'+data_item['code_v0_no_empty_lines']
        input_tokenized = tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True)
        target_tokenized = tokenizer(data_item['code_v1_no_empty_lines'], return_tensors="pt", padding="max_length", truncation=True)
        input_ids_list = input_tokenized["input_ids"].tolist()
        input_masks_list = input_tokenized["attention_mask"].tolist()

        target_ids_list = target_tokenized["input_ids"].tolist()
        target_masks_list = target_tokenized["attention_mask"].tolist()
        input_ids.append(input_ids_list[-1])
        input_masks.append(input_masks_list[-1])
        target_ids.append(target_ids_list[-1])
        target_masks.append(target_masks_list[-1])
        program_ids.append(int(data_item['problem_id'][1:]))
    '''
    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    target_ids = torch.stack(target_ids)
    target_masks = torch.stack(target_masks)
    program_ids = torch.tensor(program_ids)
    times = torch.tensor(times)
    '''
    #print("Finished {}".format(chunk_number))

    return input_ids, input_masks, target_ids, target_masks, program_ids, times, num_pass_test


def generate_dataset_from_list_multi(data_list, db, tokenizer, description, num_processes=None):

    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    program_ids = []
    times = []
    num_pass_test = 0

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print('running in {} processes'.format(num_processes))
    #print(len(data_list))
    if len(data_list)%num_processes == 0:
        length = len(data_list)//num_processes
    else:
        length = len(data_list)//num_processes + 1
    indice = [(i*length, (i+1)*length) for i in range(num_processes)]
    indice[-1] = ((num_processes-1)*length, len(data_list))
    #print(indice)
    data_chunks = [(i, data_list[index[0]:index[1]]) for i,index in enumerate(indice)]
    #partial_func = partial(process_data_chunk, tokenizer=tokenizer)
    #print(len(data_chunks[-1][1]))
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(num_processes) as pool:
        partial_func = partial(process_data_chunk, tokenizer=tokenizer)
        results = pool.map(partial_func, data_chunks)

    #print(len(results[0]))
    #print(results[0][0])
    #print(len(results[0][1]))
    #print(len(results[0][2]))
    #print(len(results[0][3]))
    #print(len(results[0][4]))
    #print(len(results[0][5]))
    # Aggregate the results from all processes
    for res in results:
        for i in range(len(res[0])):
            input_ids.append(res[0][i])
            input_masks.append(res[1][i])
            target_ids.append(res[2][i])
            target_masks.append(res[3][i])
            program_ids.append(res[4][i])
            times.append(res[5][i])
        num_pass_test += res[6]
    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    target_ids = torch.tensor(target_ids)
    target_masks = torch.tensor(target_masks)
    program_ids = torch.tensor(program_ids)
    times = torch.tensor(times)

    output_dataset = TensorDataset(input_ids,input_masks,target_ids,target_masks,program_ids,times)
    print(num_pass_test/len(output_dataset))
    print("Finish Processing {} Data".format(description))
    return output_dataset

def process_data_chunk_auto_FT(args,tokenizer):
    # This function will handle the tokenization and processing of a chunk of data
    #tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    #tokenizer.pad_token = tokenizer.eos_token
    chunk_number, data = args
    #print("Start {}".format(chunk_number))
    #print(data[0])
    code_v0s = []
    code_v1s = []
    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    program_ids = []
    times = []
    #for data_item in tqdm(data, desc='Processing {}'.format(chunk_number)):
    for data_item in data:
        code_v0s.append(data_item['code_v0_no_empty_lines'])
        code_v1s.append(data_item['code_v1_no_empty_lines'])
        codes = data_item['code_v0_no_empty_lines']
        input_data = 'Given a slower version of code:\n'+codes
        input_tokenized = tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True)
        target_tokenized = tokenizer(data_item['code_v1_no_empty_lines'], return_tensors="pt", padding="max_length", truncation=True)
        input_ids_list = input_tokenized["input_ids"].tolist()
        input_masks_list = input_tokenized["attention_mask"].tolist()

        target_ids_list = target_tokenized["input_ids"].tolist()
        target_masks_list = target_tokenized["attention_mask"].tolist()
        input_ids.append(input_ids_list[-1])
        input_masks.append(input_masks_list[-1])
        target_ids.append(target_ids_list[-1])
        target_masks.append(target_masks_list[-1])
        program_ids.append(int(data_item['problem_id'][1:]))
    '''
    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    target_ids = torch.stack(target_ids)
    target_masks = torch.stack(target_masks)
    program_ids = torch.tensor(program_ids)
    times = torch.tensor(times)
    '''
    #print("Finished {}".format(chunk_number))

    return input_ids, input_masks, target_ids, target_masks, program_ids, times

def generate_dataset_from_list_multi_auto_FT(data_list, db, tokenizer, description, num_processes=None):

    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    program_ids = []
    times = []

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print('running in {} processes'.format(num_processes))
    #print(len(data_list))
    length = len(data_list)//num_processes
    indice = [(i*length, (i+1)*length) for i in range(num_processes)]
    indice[-1] = ((num_processes-1)*length, len(data_list))
    #print(indice)
    data_chunks = [(i, data_list[index[0]:index[1]]) for i,index in enumerate(indice)]
    #partial_func = partial(process_data_chunk, tokenizer=tokenizer)
    #print(len(data_chunks[-1][1]))
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(num_processes) as pool:
        partial_func = partial(process_data_chunk_auto_FT, tokenizer=tokenizer)
        results = pool.map(partial_func, data_chunks)

    #print(len(results[0]))
    #print(results[0][0])
    #print(len(results[0][1]))
    #print(len(results[0][2]))
    #print(len(results[0][3]))
    #print(len(results[0][4]))
    #print(len(results[0][5]))
    # Aggregate the results from all processes
    for res in results:
        for i in range(len(res[0])):
            input_ids.append(res[0][i])
            input_masks.append(res[1][i])
            target_ids.append(res[2][i])
            target_masks.append(res[3][i])
            program_ids.append(res[4][i])
            times.append(res[5][i])
    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    target_ids = torch.tensor(target_ids)
    target_masks = torch.tensor(target_masks)
    program_ids = torch.tensor(program_ids)
    times = torch.tensor(times)

    output_dataset = TensorDataset(input_ids,input_masks,target_ids,target_masks,program_ids,times)
    print(len(output_dataset))
    print("Finish Processing Train Data")
    return output_dataset

def generate_dataset_from_list_auto_FT(data_list,db,tokenizer,description):
    
    input_ids = []
    input_masks = []
    target_ids = []
    target_masks = []
    
    times = []
    test = "Processing " + description + " data"
    if db:
        i = 0
    for data in tqdm(data_list, desc=test):
        code_v0=data['code_v0_no_empty_lines']
        code_v1=data['code_v1_no_empty_lines']
        
        input_data = '#slower version:\n'+code_v0  +'\n#optimized version of the same code:\n' + code_v1
        #target_data = '#slower version:\n'+code_v0  +'\n#optimized version of the same code:\n' + code_v1
        input_tokenized = tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True)
        #target_tokenized = tokenizer(target_data, return_tensors="pt", padding="max_length", truncation=True)
        input_ids.append(input_tokenized["input_ids"][-1])
        input_masks.append(input_tokenized["attention_mask"][-1])
        #target_ids.append(target_tokenized["input_ids"][-1])
        #target_masks.append(target_tokenized["attention_mask"][-1])
        
        if db and i == 32: break
        if db: i += 1
    input_ids = torch.stack(input_ids)
    input_masks = torch.stack(input_masks)
    #target_ids = torch.stack(target_ids)
    #target_masks = torch.stack(target_masks)
    
    #print(input_ids.shape)
    #print(input_masks.shape)
    #print(target_ids.shape)
    #print(target_masks.shape)
    output_dataset = TensorDataset(input_ids,input_masks)
    print(len(output_dataset))
    return output_dataset

def datahandler_auto_FT(data_args, tokenizer):
    train_data_list = read_data(data_args.train_file_path)
    
    start_time = time.time()
    train_dataset = generate_dataset_from_list_auto_FT(train_data_list,data_args.db,tokenizer,"Train")
    end_time = time.time()
    print("use {} seconds".format(end_time-start_time))
    
    return train_dataset

def datahandler(data_args, generate_args, tokenizer):
    start_time = time.time()
    train_data_list = read_data(data_args.train_file_path)
    test_data_list = read_data(data_args.test_file_path)
    vali_data_list = read_data(data_args.vali_file_path)
    train_dataset = generate_dataset_from_list_multi(train_data_list,generate_args.db,tokenizer,"Train",50)
    test_dataset = generate_dataset_from_list_multi(test_data_list,generate_args.db,tokenizer,"test",50)
    vali_dataset = generate_dataset_from_list_multi(vali_data_list,generate_args.db,tokenizer,"vali",50)
    end_time = time.time()
    print("use {} seconds".format(end_time-start_time))
    #return test_dataset,vali_dataset
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
