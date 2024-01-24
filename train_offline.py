import os
import time
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
# from tensorboardX import SummaryWriter

from model import OfflineClassifier

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

opts = argparse.ArgumentParser()
opts.add_argument("--random_seed", type=int, default=3407)
opts.add_argument("--device", type=str, default='cuda:0', help='compute device')
opts.add_argument("--data_path", type=str, default="./actions/v2_00500.npy", help="repo")
opts.add_argument("--epochs",type=int, default=1500, help="number of epochs")
opts.add_argument("--batch_size", type=int, default=10, help="batch size")
opts.add_argument("--lr", type=float, default=1e-2, help="learning rate")
opts.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
opts.add_argument("--step_size", type=int, default=100, help="learning rate decay after how much step")
opts.add_argument("--gamma", type=float, default=0.9, help="learning rate decay factor")
args = opts.parse_args()
print(args)

device = torch.device(args.device)
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)

model = OfflineClassifier()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
loss = nn.CrossEntropyLoss()

data = np.load(args.data_path)
states = torch.from_numpy(data[:, :8]).float().to(args.device)
actions = torch.from_numpy(data[:, 8]).long().to(args.device)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_scores = 0
    pred_actions = model(states)
    batch_loss = loss(pred_actions, actions)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    lr_scheduler.step()        
    print("Epoch {} with training loss {}".format(epoch, batch_loss.item()))

torch.save(model.state_dict(), "./ckpts/v3.pth")

    