import os
import time
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from model import BNN 


device = 'cuda:0'


opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--device", type=str, default='cuda:0', help='compute device')
opts.add_argument("--data_path", type=str, default="./sample2.npy", help="repo")
opts.add_argument("--epochs",type=int, default=1500, help="number of epochs")
opts.add_argument("--batch_size", type=int, default=10, help="batch size")
opts.add_argument("--lr", type=float, default=1e-2, help="learning rate")
opts.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
opts.add_argument("--step_size", type=int, default=100, help="learning rate decay after how much step")
opts.add_argument("--gamma", type=float, default=0.9, help="learning rate decay factor")
args = opts.parse_args()
print(args)

torch.manual_seed(args.random_seed)
model = BNN(9)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

data = np.load(args.data_path)
states = torch.from_numpy(data[:, :8]).float().to(device)
actions = torch.unsqueeze(torch.from_numpy(data[:, 8]).float().to(device),dim=1)
targets =torch.from_numpy(data[:, 9:]).float().to(device)

for epoch in range(1, args.epochs + 1):
    model.train()
    st=torch.cat([states, actions],dim=1)
    model.zero_grad()
    outputs = model(st, sample=True,cal=True)
    log_prior= model.log_prior()
    log_variational_posterior=model.log_posterior()
    mse_loss = model.mse_l(outputs, targets)
    loss =(log_variational_posterior - log_prior) + mse_loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if(epoch%100==0):    
        print("Epoch {} with training loss {}".format(epoch, loss.item()/data.shape[0]))

torch.save(model.state_dict(), "./ckpt/v7.pth")