import random
import numpy as np
import model
from model import params, backprop, setseed
from optimisers import ManualSGD, ManualSGDMomentum, ManualAdamW



# Do SGD
batch = 100
lr = 0.001
# opt = ManualSGD(params, lr=lr)
# opt = ManualSGDMomentum(params, lr=lr, momentum=0.9)
opt = ManualAdamW(params, lr=lr)
# opt = torch.optim.SGD(params, lr=lr)
# opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
# opt = torch.optim.AdamW(params, lr=lr)
if __name__ == "__main__":
    for epoch in range(10):
        setseed(random.randint(0, 2147483647)) # shuffle data
        for x in range(len(model.train_image) // batch):
            opt.zero_grad()
            backprop(model.train_image[x*batch:x*batch+batch], model.train_label[x*batch:x*batch+batch])
            opt.step() # SGD optimiser

    # Save the weights to a file
    np.savez('weights.npz', *[p.detach().cpu().numpy() for p in params])
    print("Training done, weights saved to weights.npz")
