import torch


class Optimiser:
    def __init__(self, params, lr):
        self.params = list(params)  # model.py weight tensors
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class ManualSGD(Optimiser):
    # torch.optim.SGD(params, lr=lr)

    def step(self):
        with torch.no_grad(): 
            for p in self.params:
                p -= self.lr * p.grad


class ManualSGDMomentum(Optimiser):
    # v <-- mu * v + \Nabla L
    # p <-- p - lr * v
    # torch.optim.SGD(params, lr=lr, momentum=)

    def __init__(self, params, lr, momentum):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad(): 
            for p, v in zip(self.params, self.velocities):
                v *= self.momentum 
                v += p.grad             # v <-- mu * v + g
                p -= self.lr * v        # p <-- p - lr * v


class ManualAdamW(Optimiser):
    # t <-- t + 1
    # p <-- p * (1 - lr * weight_decay)                 
    # m <-- beta1 * m + (1 - beta1) * g
    # v <-- beta2 * v + (1 - beta2) * g**2
    # m_hat = m / (1 - beta1**t),  v_hat = v / (1 - beta2**t)
    # p <-- p - lr * m_hat / (sqrt(v_hat) + eps)   
    #torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.ms = [torch.zeros_like(p) for p in self.params]  # 1st moment (mean of g)
        self.vs = [torch.zeros_like(p) for p in self.params]  # 2nd moment (mean of g**2)

    def step(self):
        self.t += 1                                   # once per step, shared by all params
        bias1 = 1 - self.beta1 ** self.t
        bias2 = 1 - self.beta2 ** self.t
        with torch.no_grad():
            for p, m, v in zip(self.params, self.ms, self.vs):
                g = p.grad
                p *= 1 - self.lr * self.weight_decay  # p <-- p * (1 - lr * wd)
                m *= self.beta1
                m += (1 - self.beta1) * g             # m <-- b1 * m + (1 - b1) * g
                v *= self.beta2
                v += (1 - self.beta2) * g * g         # v <-- b2 * v + (1 - b2) * g**2
                m_hat = m / bias1
                v_hat = v / bias2
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
            