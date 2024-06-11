import torch
from transformers import GPT2Tokenizer,BartTokenizer
import jsonlines
import pickle
from nltk import bleu
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import pdb
import json
from tqdm import trange
import datetime
import random
from datasets import load_dataset
def test_algorithm(hps,model,tokenizer,wandb):
    dir='./tokenize/algorithm/'
    names=["Backtracking","Binary_Search_Tree","Binary_Search","Binary_Tree","Breadth-First_Search","Depth-First_Search","Divide_and_Conquer","Dynamic_Programming","Graph","Greedy","Heap_(Priority_Queue)","Ordered_Set","Recursion","Sorting","Stack","Tree","Two_Pointers"]
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                with open("./out3/"+name,'w')as f:
                    f.write(json.dumps(nlls))
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})
def test_wiki(hps,model,tokenizer,wandb):
    dir='./tokenize/other/'
    names=['biology','chemistry','computer_science','economics','law','linguistics','psychology','physics']
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})

def test_all(hps,model,tokenizer,wandb):
    dir='./tokenize/all/'
    names=['arxivs','books','c4s','githubs','stackexchanges','wikis']
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})
def test_stackexchange(hps,model,tokenizer,wandb):
    dir='./tokenize/stackexchange/'
    names=['0','1','2','3','4']
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})

def test_code(hps,model,tokenizer,wandb):
    dir='./tokenize/code/'
    names=['C','C++','CSS','HTML','Java','Python','PHP','JavaScript','Shell','R','Web_Ontology_Language','SQL','TeX']
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})

def test_arxiv(hps,model,tokenizer,wandb):
    dir='./tokenize/arxiv/'
    # names=['C','C++','CSS','HTML','Java','Python','PHP','JavaScript','Shell','R','Web_Ontology_Language','SQL','TeX']
    names=['Physics','Mathematics','Computer_Science','Quantitative_Biology','Quantitative_Finance','Statistics','Electrical_Engineering_and_Systems_Science','Economics']
    model.eval()     
    with torch.no_grad():
        for name in names:
            with open(dir+name+'.pkl','rb') as f1:
                test_data_output=pickle.load(f1)
                data_input_ids, data_attention_mask=test_data_output['input_ids'][:,:],test_data_output['attention_mask'][:,:]
                del test_data_output
            Data = TensorDataset(data_input_ids, data_attention_mask)
            data_sampler = torch.utils.data.distributed.DistributedSampler(Data, shuffle=True)
            data_dataloader = DataLoader(Data, batch_size=hps.per_device_train_batch_size, sampler=data_sampler,shuffle=hps.shuffle, drop_last=False)
            torch.distributed.barrier() 
            nlls=[]
            epoch_step=0
            total_loss=0
            if hps.global_rank == 0:
                data_dataloader = tqdm(data_dataloader)
                data_dataloader.set_description(name)
            for batch_data in data_dataloader:
                input_ids,attention_mask=batch_data
                labels = input_ids.clone()
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                if hps.use_gpu:
                    input_ids=input_ids.cuda()
                    attention_mask=attention_mask.cuda()
                    labels=labels.cuda()
                output = model(input_ids=input_ids, labels=labels,attention_mask=attention_mask)
                loss_sum = torch.tensor([output.loss.item()], device='cuda')
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                now_loss = loss_sum.item()/dist.get_world_size()
                total_loss += now_loss.item()
                nlls.append(now_loss)
                epoch_step=epoch_step+1
                if hps.global_rank == 0:
                    tqdm.write(f'loss: {now_loss:.4f}')
                    tqdm.write(f'total_loss: {total_loss/epoch_step:.4f}')
            if hps.global_rank == 0:
                ppl=torch.exp(torch.stack(nlls).mean())
                if hps.use_wandb:
                    wandb.log({"{}_loss".format(name):total_loss/epoch_step, 
                           "{}_ppl".format(name):ppl})

def test(hps,model,tokenizer,forget_dev_dataloader,other_forget_dev_data):
    other_forget_dev_data_loss=[]
    ppls=[]
    print("====测试集测试开始====")
    t=trange(len(forget_dev_dataloader))
    dev_loss=0
    dev_step=0
    nlls=[]
    for i ,batch in zip(t,forget_dev_dataloader):
        if hps.use_gpu:
                batch = tuple(term.cuda() for term in batch)
        input_id,mask=batch
        label = input_id.clone()
        label[label[:, :] == tokenizer.pad_token_id] = -100
        loss=model(input_ids=input_id, labels=label,attention_mask=mask)['loss']
        dist.reduce(loss, 0)
        loss=loss/dist.get_world_size()
        dev_loss+=loss.item()
        nlls.append(loss)
        dev_step=dev_step+1
        t.set_postfix(total_loss='{}'.format(dev_loss/dev_step))
    dev_ppl = torch.exp(torch.stack(nlls).mean())
    if hps.global_rank == 0:
        print("[{} Loss]: {}".format(hps.forget,dev_loss/dev_step))
        print("[{} PPL]: {}".format(hps.forget,dev_ppl))
    for other_forget_dev in other_forget_dev_data:
        total_loss = 0
        test_step=0
        nlls=[]
        t = trange(len(other_forget_dev))
        for i, batch in zip(t, other_forget_dev):
            if hps.use_gpu:
                batch = tuple(term.cuda() for term in batch)
            input_id,mask=batch
            label = input_id.clone()
            label[label[:, :] == tokenizer.pad_token_id] = -100
            loss=model(input_ids=input_id, labels=label,attention_mask=mask)['loss']
            dist.barrier()
            dist.reduce(loss, 0)
            total_loss+=loss.item()
            nlls.append(loss)
            test_step=test_step+1
            t.set_postfix(total_loss='{}'.format(total_loss/test_step))
        other_forget_dev_data_loss.append(total_loss/test_step)
        ppl = torch.exp(torch.stack(nlls).mean())
        ppls.append(ppl.item())
    # wandb.log({"math Loss":dev_loss/dev_step,"math PPL":dev_ppl})
    if hps.global_rank == 0:
        for i in range(len(other_forget_dev_data)):
        # wandb.log({"{} Loss".format(hps.like_forget[i]):dev_loss/dev_step,"[{} PPL]: {}".format(hps.like_forget[i]):other_forget_dev_data_loss[i]})
            print("[{} Loss]: {}".format(hps.like_forget[i],other_forget_dev_data_loss[i]))
            print("[{} PPL]: {}".format(hps.like_forget[i],ppls[i]))
    return dev_loss/dev_step,dev_ppl,other_forget_dev_data_loss,ppls
def dev_code(hps,model,tokenizer,dev_dataloader):
    dev_total_loss=0
    if hps.global_rank == 0:
        print("====验证集测试开始====")
        dev_dataloader = tqdm(dev_dataloader,leave=False)
        dev_dataloader.set_description("Study")
    dev_step=0
    nlls=[]
    for batch in dev_dataloader:
        if hps.use_gpu:
            batch = tuple(term.cuda() for term in batch)
        input_id,mask=batch
        mask=mask
        input_id=input_id
        label = input_id.clone()
        label[label[:, :] == tokenizer.pad_token_id] = -100
        loss=model(input_ids=input_id, labels=label,attention_mask=mask)['loss']
        # dist.barrier()
        dist.reduce(loss, 0)
        loss = loss / dist.get_world_size()
        nlls.append(loss)
        dev_total_loss+=loss.item()
        tqdm.write(f'loss: {loss.item():.4f}')
        tqdm.write(f'total_loss: {dev_total_loss/(dev_step+1):.4f}')
        dev_step=dev_step+1
    dev_ppl = torch.exp(torch.stack(nlls).mean())
    if hps.global_rank == 0:
        print("[dev_dev Loss]: {}".format(dev_total_loss/dev_step))
        print("[dev_dev PPL]: {}".format(dev_ppl))
    return dev_total_loss/dev_step,dev_ppl
def dev(hps,model,tokenizer,forget_dev_dataloader,dev_dataloader):
    forget_dev_total_loss=0
    t = trange(len(forget_dev_dataloader))
    forget_step=0
    nlls=[]
    print("====验证集测试开始====")
    for i, batch in zip(t, forget_dev_dataloader):
        if hps.use_gpu:
            batch = tuple(term.cuda() for term in batch)
        input_id,mask=batch
        mask=mask
        input_id=input_id
        label = input_id.clone()
        label[label[:, :] == tokenizer.pad_token_id] = -100
        loss=model(input_ids=input_id, labels=label,attention_mask=mask)['loss']
        # dist.barrier()
        dist.reduce(loss, 0)
        loss = loss / dist.get_world_size()
        nlls.append(loss)
        forget_dev_total_loss+=loss.item()
        t.set_postfix(forget_dev_loss='{}'.format(forget_dev_total_loss/(forget_step+1)))
        forget_step=forget_step+1
    forget_ppl = torch.exp(torch.stack(nlls).mean())
    dev_total_loss=0
    dev_step=0
    nlls=[]
    t = trange(len(dev_dataloader))
    for i, batch in zip(t, dev_dataloader):
        if hps.use_gpu:
            batch = tuple(term.cuda() for term in batch)
        input_id,mask=batch
        mask=mask
        input_id=input_id
        label = input_id.clone()
        label[label[:, :] == tokenizer.pad_token_id] = -100
        loss=model(input_ids=input_id, labels=label,attention_mask=mask)['loss']
        # dist.barrier()
        dist.reduce(loss, 0)
        loss = loss / dist.get_world_size()
        nlls.append(loss)
        dev_total_loss+=loss.item()
        t.set_postfix(dev_loss='{}'.format(dev_total_loss/(dev_step+1)))
        dev_step=dev_step+1
    dev_ppl = torch.exp(torch.stack(nlls).mean())
    # wandb.log({"forget_dev Loss":forget_dev_total_loss/forget_step,"forget_dev PPL":forget_ppl,"dev_dev Loss":dev_total_loss/dev_step,"dev_dev PPL":dev_ppl})
    if hps.global_rank == 0:
        print("[forget_dev Loss]: {}".format(forget_dev_total_loss/forget_step))
        print("[forget_dev PPL]: {}".format(forget_ppl))
        print("[dev_dev Loss]: {}".format(dev_total_loss/dev_step))
        print("[dev_dev PPL]: {}".format(dev_ppl))
    return forget_dev_total_loss/forget_step,forget_ppl,dev_total_loss/dev_step,dev_ppl
    #result=validation_forget(hps,model,tokenizer,forget_dev_dataloader,logger,'target')

       

def tokenize_predata(datas,tokenizer):
    outputs = tokenizer(datas, padding="max_length",truncation =True,max_length=1024,return_tensors='pt')
    return outputs

def read_predata(path):
    with open(path,'r') as f:
        predata=json.loads(f.read())
    random.shuffle(predata)
    # result=[]
    # for data in predata:
    #     if len(data)<=1000:
    #         result.append(data)
    #     else:
    #         while(len(data)>1000):
    #             result.append(data[:1000])
    #             data=data[1000:]
    #         result.append(data)
    return predata



