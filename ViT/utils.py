import torch 
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from topk_accuracy import accuracy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def eval_fn(model, loss_fn, valloader, device, epoch):
    model.eval()
    val_loss, val_accu = 0.0, 0.0
    tester = tqdm(valloader, unit="batch")
    tester.set_description(f"Epoch {epoch}-Valid")
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(tester):
            idx += 1
            x = x.to(device)
            y = y.to(device)

            cls = model(x) 
            loss = loss_fn(cls, y)

            val_loss += loss.item() * y.size(0)
            val_accu += accuracy(cls, y, topk=5)

            tester.set_postfix(loss = val_loss / idx, accuracy = val_accu / idx)
    
    return val_loss, val_accu


def train_fn(model, loss_fn, optim, lr_sch, trainloader, valloader, device, epochs=10, step_type="accuracy"):
    model.train()
    
    train_hist, val_hist = [], []
    
    for i in range(1, epochs + 1):
        val_loss, val_accu = 0.0, 0.0
        train_loss, train_accu = 0.0, 0.0

        trainer = tqdm(trainloader, unit="batch")
        trainer.set_description(f"Epoch {i}-Train")

        for idx, (x, y) in enumerate(trainer):
            idx += 1
            x = x.to(device)
            y = y.to(device)

            cls = model(x)
            loss = loss_fn(cls, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()

            train_loss += loss.item() * y.size(0)
            train_accu += accuracy(cls, y, topk=5)

            trainer.set_postfix(loss = train_loss / idx, accuracy = train_accu / idx)

        vloss, vaccu = eval_fn(model, loss_fn, valloader, device, i)
        val_loss += vloss
        val_accu += vaccu
            
        if step_type == "accuracy":
            lr_sch.step(val_accu)
        elif step_type == "loss":
            lr_sch.step(val_loss)
        else:
            lr_sch.step()
            
        val_hist.append(val_accu / len(valloader))
        train_hist.append(train_accu / len(trainloader))
    
    return model, optim, train_hist, val_hist