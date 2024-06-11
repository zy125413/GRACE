import torch
import math
import wandb
import pickle
import json
import time
import torch.distributed as dist
import deepspeed
from torch.utils.data import TensorDataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
from tools import dev, test_code

def hyper_parameters():
    """
    Parse and return hyperparameters from command line arguments.
    """
    parser = ArgumentParser(description='llama-7B')

    # Model and tokenizer directories
    parser.add_argument('--model_dir', type=str, default='./llama-2-7b-hf')
    parser.add_argument('--tokenizer_dir', type=str, default='./llama-2-7b-hf')

    # Data directories
    parser.add_argument('--forget_dir', type=str, default='./tokenize/code/')
    parser.add_argument('--dev_dir', type=str, default='./tokenize/code_dev/')
    parser.add_argument('--test_dir', type=str, default='./code_test/')

    # Save directory and experiment settings
    parser.add_argument('--save_dir', type=str, default='./save_model/')
    parser.add_argument('--name', type=str, default='C')
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--evaluation_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=3184)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--exp_name", type=str, default="data")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)

    # Add DeepSpeed arguments
    deepspeed.add_config_arguments(parser)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':    

    # Parse hyperparameters
    hps = hyper_parameters()
    # Initialize distributed training with DeepSpeed
    deepspeed.init_distributed()
    hps.global_rank = torch.distributed.get_rank()
    print(hps)

    # Set random seeds for reproducibility
    torch.manual_seed(hps.seed)
    torch.cuda.manual_seed(hps.seed)

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(hps.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        hps.model_dir,
        low_cpu_mem_usage=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Set device for local rank
    local_rank = hps.local_rank
    torch.cuda.set_device(local_rank)

    # Load DeepSpeed configuration
    with open("./ds_config2.json", 'r') as f:
        ds_config = json.loads(f.read())
    ds_config['optimizer']['params']['lr'] = hps.learning_rate
    ds_config['train_micro_batch_size_per_gpu'] = hps.per_device_train_batch_size
    ds_config['gradient_accumulation_steps'] = hps.gradient_accumulation_steps

    # Load data configuration
    with open("./code.json", 'r') as f:
        data = json.loads(f.read())
    hps.dev_loss = data[hps.name]["dev_loss"]  # non-target loss
    hps.unlearn_loss = data[hps.name]["random_loss"]  # random loss 

    # Initialize Weights & Biases for logging if enabled
    if hps.use_wandb and hps.global_rank == 0:
        wandb.init(
            project=hps.exp_name,
            config={
                "learning_rate": hps.learning_rate,
                "name": hps.name,
                "epochs": hps.num_train_epochs,
                "train_batch_size": dist.get_world_size() * hps.per_device_train_batch_size * hps.gradient_accumulation_steps,
                "per_device_train_batch_size": hps.per_device_train_batch_size,
                "accumulation steps": hps.gradient_accumulation_steps,
                "num_gpus": dist.get_world_size()
            }
        )

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=hps, config=ds_config,
        model=model, model_parameters=model.parameters()
    )

    # Load data for forgetting, studying, and validation
    with open(hps.forget_dir + hps.name + '.pkl', "rb") as f:
        forget_output = pickle.load(f)
        forget_ids, forget_mask = forget_output['input_ids'][:, :], forget_output['attention_mask'][:, :]
        del forget_output

    with open(hps.dev_dir + hps.name + '.pkl', "rb") as f:
        study_output = pickle.load(f)
        study_ids, study_mask = study_output['input_ids'][:, :], study_output['attention_mask'][:, :]
        del study_output

    with open(hps.test_dir + hps.name + '.pkl', "rb") as f:
        dev_output = pickle.load(f)
        dev_ids, dev_mask = dev_output['input_ids'][:, :], dev_output['attention_mask'][:, :]
        del dev_output

    # Create data loaders
    Forget = TensorDataset(forget_ids, forget_mask)
    Forget_sampler = torch.utils.data.distributed.DistributedSampler(Forget, shuffle=True)
    forget_dataloader = DataLoader(Forget, batch_size=hps.per_device_train_batch_size, sampler=Forget_sampler, shuffle=hps.shuffle, drop_last=False)

    study_dataloader = []
    for i in range(math.floor(len(study_ids) / 2000)):
        temp = TensorDataset(study_ids[i * 2000:(i + 1) * 2000, :], study_mask[i * 2000:(i + 1) * 2000, :])
        temp_sampler = torch.utils.data.distributed.DistributedSampler(temp, shuffle=True)
        study_dataloader.append(DataLoader(temp, batch_size=hps.per_device_train_batch_size, sampler=temp_sampler, shuffle=hps.shuffle, drop_last=False))

    Dev = TensorDataset(dev_ids, dev_mask)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(Dev, shuffle=True)
    dev_dataloader = DataLoader(Dev, batch_size=hps.per_device_train_batch_size, sampler=dev_sampler, shuffle=hps.shuffle, drop_last=False)

    global_step = hps.init_global_step
    print("[INFO] Start Training")
    loss_stack = []
    patient = 0
    best_accuracy = 0
    keep_ls = []
    stop_train = False
    start_time = time.time()
    dist.barrier()
    study_index = 0

    # Training loop
    for epoch in range(hps.num_train_epochs):
        print('[Epoch] {}'.format(epoch))
        epoch_step = 0
        total_loss = 0
        patient = 0
        Forget = True
        forget_dataloader.sampler.set_epoch(epoch)
        if hps.global_rank == 0:
            pbar = tqdm(total=len(forget_dataloader), desc=hps.name)
        for batch_data in forget_dataloader:
            model_engine.train()
            input_ids, attention_masks = batch_data
            labels = input_ids.clone()
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            if hps.use_gpu:
                input_ids = input_ids.to(local_rank)
                attention_masks = attention_masks.to(local_rank)
                labels = labels.to(local_rank)
            output = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_masks)
            loss = -output['loss']
            model_engine.backward(loss)
            if hps.global_rank == 0:
                if hps.use_wandb:
                    wandb.log({
                        "optimizer._global_grad_norm": model_engine.optimizer._global_grad_norm,
                    }, step=global_step)
            if model_engine.optimizer._global_grad_norm < 2000:
                keep_ls.append({'input_ids': input_ids.detach().cpu(), 'attention_mask': attention_masks.detach().cpu()})
            model_engine.step()
            loss_sum = torch.tensor([loss.item()], device='cuda')
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            loss_avg = loss_sum.item() / dist.get_world_size()
            mem = torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024**3

            if hps.global_rank == 0:
                if hps.use_wandb:
                    wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "Batch Step": epoch_step, "Global Step": epoch * len(forget_dataloader) + epoch_step, "Loss": loss_avg, "memory_stats": mem}, step=global_step)

            total_loss += loss_avg
            if hps.global_rank == 0:
                pbar.set_postfix({"total_loss": total_loss / (epoch_step + 1), "loss": (loss_avg)})
                pbar.update(1)

            if Forget:
                dev_total_loss = 0
                study_steps = 0
                model_engine.train()
                model_engine.zero_grad()
                study_dataloader[study_index].sampler.set_epoch(study_index)
                if hps.global_rank == 0:
                    print("====开始补充其他知识=====")
                    study_pbar = tqdm(total=len(study_dataloader[study_index]), desc="Study", leave=False)

                for batch_study_data in study_dataloader[study_index]:
                    input_ids, attention_masks = batch_study_data
                    labels = input_ids.clone()
                    labels[labels[:, :] == tokenizer.pad_token_id] = -100
                    if hps.use_gpu:
                        input_ids = input_ids.to(local_rank)
                        attention_masks = attention_masks.to(local_rank)
                        labels = labels.to(local_rank)
                    study_output = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_masks)
                    study_loss = study_output['loss']
                    model_engine.backward(study_loss)
                    model_engine.step()
                    study_loss_sum = torch.tensor([study_loss.item()], device='cuda')
                    dist.all_reduce(study_loss_sum, op=dist.ReduceOp.SUM)
                    study_loss_avg = study_loss_sum.item() / dist.get_world_size()
                    dev_total_loss += study_loss_avg
                    study_steps += 1
                    if hps.global_rank == 0:
                        study_pbar.set_postfix({"total_loss": dev_total_loss / study_steps, "loss": (study_loss_avg)})
                        study_pbar.update(1)
                Forget = False
                study_index += 1
                if hps.global_rank == 0:
                    study_pbar.close()
                    print("====补充训练完成====")
                model_engine.eval()
                with torch.no_grad():
                    forget_loss, forget_ppl, dev_loss, dev_ppl = dev(hps, model_engine, tokenizer, forget_dataloader, dev_dataloader)
                if hps.global_rank == 0:
                    if hps.use_wandb:
                        wandb.log({
                            "after_forget_loss": forget_loss, "after_forget_ppl": forget_ppl, "after_dev_loss": dev_loss, "after_dev_ppl": dev_ppl
                        }, step=global_step)
                if dev_loss > hps.dev_loss:
                    print("step{} Model is forget".format(global_step))
                    Forget = True

            if global_step % hps.evaluation_steps == 0 and global_step != 0:
                print("====开始验证=====")
                model_engine.eval()
                with torch.no_grad():
                    forget_loss, forget_ppl, dev_loss, dev_ppl = dev(hps, model_engine, tokenizer, forget_dataloader, dev_dataloader)
                    if hps.global_rank == 0:
                        if hps.use_wandb:
                            wandb.log({
                                "forget_loss": forget_loss, "dev_loss": dev_loss, "forget_ppl": forget_ppl, "dev_ppl": dev_ppl
                            }, step=global_step)
                if hps.global_rank == 0:
                    if dev_loss > hps.dev_loss:
                        print("Step {} Stopping Training is Forget".format(global_step))
                        Forget = True

                    if forget_loss > hps.unlearn_loss:
                        print("Step {} Stopping Training by Early Stopping".format(global_step))
                        stop_train = True
            epoch_step += 1
            global_step += 1
            if stop_train:
                break
        if stop_train:
            break

    model_engine.eval()
    if hps.global_rank == 0:
        pbar.close()
    with torch.no_grad():
        test_code(hps, model, tokenizer, wandb)
    model_engine.save_pretrained(hps.save_dir + hps.name)
