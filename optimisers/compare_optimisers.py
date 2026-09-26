import numpy as np
import torch
import model
from model import params, backprop, forward_pass, setseed
from optimisers import ManualSGD, ManualSGDMomentum, ManualAdamW


batch = 100
epochs = 10
runs = 10

# Each method gets the lr that works for it
methods = {
    "ManualSGD":                 lambda: ManualSGD(params, lr=0.1),
    "torch.optim.SGD":           lambda: torch.optim.SGD(params, lr=0.1),
    "ManualSGDMomentum":         lambda: ManualSGDMomentum(params, lr=0.01, momentum=0.9),
    "torch.optim.SGD (mu=0.9)":  lambda: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    "ManualAdamW":               lambda: ManualAdamW(params, lr=0.001),
    "torch.optim.AdamW":         lambda: torch.optim.AdamW(params, lr=0.001),
}

# Kaiming initialisation again
def reset_params(seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in params:
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * np.sqrt(2 / p.shape[0]))
            else:
                p.zero_()

def success_rate():
    with torch.no_grad():
        predictions = forward_pass(model.test_image).argmax(dim=1)
        return (predictions == model.test_label).float().mean().item() * 100

results = {}
for name, make_opt in methods.items():
    rates = []
    for run in range(runs):
        reset_params(run)              # run r starts from the same weights for every method
        opt = make_opt()
        for i in range(epochs):
            setseed(run * epochs + i)  # ...and sees the same shuffle order
            for x in range(len(model.train_image) // batch):
                opt.zero_grad()
                backprop(model.train_image[x*batch:x*batch+batch], model.train_label[x*batch:x*batch+batch])
                opt.step()
        rates.append(success_rate())
        print(f"{name:26s} run {run}: {rates[-1]:.2f}%", flush=True)
    results[name] = rates

print()
for name, rates in results.items():
    print(f"{name:26s} ({np.mean(rates):.2f} ± {np.std(rates, ddof=1):.2f})%")  # sample std over runs
