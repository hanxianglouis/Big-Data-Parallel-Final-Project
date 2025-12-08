import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import time
import tqdm
import os
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Union

from preprocessing import preprocessing
from dataset import CriteoDataset
from model import EcomDFCL
from loss_functions import local_prediction_loss, decision_policy_learning_loss

def validate(model: Union[EcomDFCL, DDP], val_loader: DataLoader, device, alpha=1) :
    model.eval()
    val_losses = []
    for features, treatment, cost, revenue in val_loader :
        features, treatment, cost, revenue = features.to(device), treatment.to(device), cost.to(device), revenue.to(device)
        output = model(features)
        L_pred = local_prediction_loss(output,treatment,cost,revenue)
        L_decision = decision_policy_learning_loss(output,treatment,cost,revenue,device)

        loss = alpha * L_pred - L_decision
        val_losses.append(loss.item())

    return np.sum(val_losses)/len(val_loader)

def train(model: Union[EcomDFCL, DDP], train_loader: DataLoader,val_loader: DataLoader, train_sampler: DistributedSampler, alpha=1, epochs=10, lr=0.001, patience=10, device='cuda') :
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = 100000000
    no_imporovement = 0
    os.makedirs("model", exist_ok=True)
    rank = dist.get_rank()

    for epoch in range(epochs) :
        train_sampler.set_epoch(epoch)
        model.train()
        train_losses = []
        t0 = time.time()
        for features, treatment, cost, revenue in tqdm.tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}") :

            features, treatment, cost, revenue = features.to(device), treatment.to(device), cost.to(device), revenue.to(device)
            output = model(features)
            L_pred = local_prediction_loss(output,treatment,cost,revenue)
            L_decision = decision_policy_learning_loss(output,treatment,cost,revenue,device)
            loss = alpha * L_pred - L_decision

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        train_loss = np.sum(train_losses)/len(train_loader)
        val_loss = validate(model,val_loader,device,alpha)
        if rank == 0:
            print(f"[Epoch {epoch+1}] train loss: {train_loss:.4f}, validate loss: {val_loss:.4f}, time consuming: {time.time()-t0}s")

        if best_val_loss > val_loss :
            no_imporovement = 0
            best_val_loss = val_loss
        else :
            no_imporovement +=1
            print(f"No imporvement for {no_imporovement} epoch(s)")

        if no_imporovement >= patience :
            if rank == 0:
                print(f"Exceed the patience: {patience} epochs, early stop!")
            break
        
    return model

def main() :
    if torch.cuda.is_available() :
        device = 'cuda'
        backend = "nccl"
    else :
        device = 'cpu'
        backend = "gloo"

    
    dist.init_process_group(backend)
    print(
        f"[Process Info] PID={os.getpid()}, "
        f"RANK={dist.get_rank()}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
    )

    os.makedirs("data", exist_ok=True)

    raw_train_path = "data/criteo_train.csv"
    raw_val_path = "data/criteo_val.csv"
    export_train_path = "data/criteo_train.parquet"
    export_val_path = "data/criteo_val.parquet"

    rank = dist.get_rank()

    if rank == 0:
        print(f"Using device: {device}")
        preprocessing(raw_train_path,raw_val_path,export_train_path,export_val_path)
        
    dist.barrier()   # wait for rank 0 to finish preprocessing

    batch_size = 256

    train_set = CriteoDataset("data/criteo_train.parquet")
    val_set = CriteoDataset("data/criteo_val.parquet")

    train_sampler = DistributedSampler(train_set,shuffle=True,drop_last=False)
    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            sampler=train_sampler,
                            num_workers=4)
    
    val_sampler = DistributedSampler(val_set,shuffle=False,drop_last=False)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=4)

    model = EcomDFCL()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    model = EcomDFCL().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = model.to(device)
    
    best_model = train(model,train_loader,val_loader,train_sampler, epochs=1, patience=10,device=device)

    final_train_loss = validate(best_model,train_loader, device=device)
    final_val_loss = validate(best_model,val_loader, device=device)

    print(f"For the best model, the train loss is {final_train_loss:.4f}, the validate loss is {final_val_loss}")

    # save the metrics
    os.makedirs("result", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f"result/result_{timestamp}.txt",'w') as f:
        f.write(f"Train Loss: {final_train_loss:.4f}\n")
        f.write(f"Validate Loss: {final_val_loss:.4f}")



if __name__ == '__main__' :
    main()