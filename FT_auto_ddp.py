
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from util import datahandler, datahandler_auto_FT
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from itertools import cycle
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

def fine_tune(args,tokenizer,train_dataset):
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')

    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
    if args.continues:
        model.load_state_dict(torch.load(args.load_from_model_path))

    device = torch.device(f"cuda:{torch.distributed.get_rank()}")

    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()])

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.fine_tuning_steps)

    model.train()

    # Use DistributedSampler for distributed data loading
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    train_dataloader = cycle(train_dataloader)

    best_train_lost = 1000

    train_sampler.set_epoch(step)
    for step in range(args.fine_tuning_steps):
        # Set the epoch for the sampler
        input_ids, input_masks, target_ids, target_masks = [t.to(device) for t in next(train_dataloader)]
        output = model(input_ids=input_ids, attention_mask=input_masks, labels=target_ids)
        loss = output.loss
        loss.backward()

        if step % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if loss < best_train_lost:
                # Only save the model from the first process to avoid overwriting
                if dist.get_rank() == 0:
                    torch.save(model.module.state_dict(), args.save_to_model_path)
                    best_train_lost = loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default='./data/python_splits/train.jsonl', type=str, help="path for data")  
    parser.add_argument("--batch_size", default=2 , type=int, help="batch size")  
    parser.add_argument("--fine_tuning_steps", default=100 , type=int, help="fine tuning steps")  
    parser.add_argument("--save_to_model_path", default='./model_param/model_auto.pth' , type=str, help="save to model path")  
    parser.add_argument("--db", action="store_true", help="debug mode")  
    parser.add_argument("--continues", action="store_true", help="continue fine tuning")  
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = datahandler_auto_FT(args, tokenizer)
    fine_tune(args,tokenizer,train_dataset)
