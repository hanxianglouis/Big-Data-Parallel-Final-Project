import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import optim
import numpy as np
import time
import tqdm
import os
from datetime import datetime

from preprocessing import preprocessing
from dataset import CriteoDataset
from model import EcomDFCL
from loss_functions import local_prediction_loss, decision_policy_learning_loss

def validate(model: EcomDFCL, val_loader: DataLoader, device, alpha=1) :
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

def train(model: EcomDFCL, train_loader: DataLoader,val_loader: DataLoader, alpha=1, epochs=10, lr=0.001, patience=10, device='cuda') :

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = 100000000
    no_imporovement = 0
    os.makedirs("model", exist_ok=True)

    for epoch in range(epochs) :
        model.train()
        train_losses = []
        t0 = time.time()
        for features, treatment, cost, revenue in tqdm.tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}") :

            features, treatment, cost, revenue = features.to(device), treatment.to(device), cost.to(device), revenue.to(device)
            output = model(features)
            L_pred = local_prediction_loss(output,treatment,cost,revenue)
            L_decision = decision_policy_learning_loss(output,treatment,cost,revenue,device)
            
            print(L_pred.item(),L_decision.item())
            loss = alpha * L_pred - L_decision

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        train_loss = np.sum(train_losses)/len(train_loader)
        val_loss = validate(model,val_loader,device,alpha)
        print(f"[Epoch {epoch+1}] train loss: {train_loss:.4f}, validate loss: {val_loss:.4f}, time consuming: {time.time()-t0}s")

        if best_val_loss > val_loss :
            no_imporovement = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"model/best_model.pth")
        else :
            no_imporovement +=1
            print(f"No imporvement for {no_imporovement} epoch(s)")

        if no_imporovement >= patience :
            print(f"Exceed the patience: {patience} epochs, early stop!")
            break

    best_model = EcomDFCL()
    best_model.load_state_dict(torch.load("model/best_model.pth"))
    best_model.to(device)  
    best_model.eval() 

    return best_model

def main() :
    if torch.cuda.is_available() :
        device = 'cuda'
    elif torch.backends.mps.is_available() :
        device = 'mps'
    else :
        device = 'cpu'

    os.makedirs("data", exist_ok=True)

    raw_train_path = "data/criteo_train.csv"
    raw_val_path = "data/criteo_val.csv"
    export_train_path = "data/criteo_train.parquet"
    export_val_path = "data/criteo_val.parquet"

    preprocessing(raw_train_path,raw_val_path,export_train_path,export_val_path)
    
    batch_size = 1024

    train_set = CriteoDataset("data/criteo_train.parquet")
    val_set = CriteoDataset("data/criteo_val.parquet")

    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=4)

    model = EcomDFCL()
    print(summary(model, (12,), batch_size=batch_size, device='cpu')) # torchsummary does not support mps, to avoid error, we use cpu temporarily
    model = model.to(device)
    
    best_model = train(model,train_loader,val_loader,epochs=1, patience=10,device=device)

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