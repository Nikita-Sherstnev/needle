"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, device=None):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.u = {w: ndl.init.zeros(*w.shape, device=device) for w in params}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data + self.weight_decay * w.data
            self.u[w].data = self.momentum * self.u[w].data + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[w].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        device=None
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        self.m = {w: ndl.init.zeros(*w.shape, device=device) for w in params}
        self.v = {w: ndl.init.zeros(*w.shape, device=device) for w in params}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data + self.weight_decay * w.data
            self.m[w].data = self.beta1 * self.m[w].data + (1 - self.beta1) * grad
            self.v[w].data = self.beta2 * self.v[w].data + (1 - self.beta2) * (grad ** 2)
            m = self.m[w].data / (1 - self.beta1 ** self.t)
            v = self.v[w].data / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * m / (v ** 0.5 + self.eps)
        ### END YOUR SOLUTION
