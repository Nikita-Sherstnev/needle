"""The module.
"""
from typing import List
from needle.autograd import Tensor
from functools import reduce
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features), dtype=dtype, device=device)
        if bias:
            self.bias = Parameter(init.kaiming_uniform(self.out_features, 1).transpose(), dtype=dtype, device=device)
        else:
            self.bias = Parameter(init.zeros(self.out_features, 1).transpose(), dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        weight = X @ self.weight
        return weight + self.bias.reshape([1 for dim in weight.shape[:-1]] + [weight.shape[-1]]).broadcast_to(weight.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], reduce(lambda a, b: a*b, X.shape[1:])))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(ops.negate(x))) ** -1
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y, dtype=logits.dtype, device=logits.device)
        Zy = ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(ops.logsumexp(logits, axes=(1,)) - Zy) / logits.shape[0]
        ### END YOUR SOLUTION


class BinaryCrossEntropy(Module):
    def forward(self, p, t):
        if len(p.shape) != len(t.shape):
            t = t.reshape(*p.shape)

        p = ops.clip(p, 1e-9, 0.999)

        tlog_p = t * ops.log(p) + (1 - t) * ops.log(1 - p)
        y = -1 * ops.summation(tlog_p) / t.shape[0]
        return y


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), dtype=dtype, device=device)
        self.bias = Parameter(init.zeros(dim, 1), dtype=dtype, device=device)
        self.running_mean = Tensor(init.zeros(dim), dtype=dtype, device=device, requires_grad=False)
        self.running_var = Tensor(init.ones(dim), dtype=dtype, device=device, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean_res = mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            var = ops.summation(((x - mean_res) ** 2), axes=(0,)) / x.shape[0]

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            var = var.reshape((1, x.shape[1])).broadcast_to(x.shape)

            norm = (x - mean_res) / ((var + self.eps) ** 0.5)
            weight = self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm
            return weight + self.bias.reshape((1, weight.shape[1])).broadcast_to(weight.shape)
        else:
            return ((x - self.running_mean) / (self.running_var + self.eps) ** 0.5)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), dtype=dtype, device=device)
        self.bias = Parameter(init.zeros(dim, 1), dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)) / x.shape[1]
        mean = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        var = ops.summation(((x - mean) ** 2), axes=(1,)) / x.shape[1]
        var = var.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        norm = (x - mean) / ((var + self.eps) ** 0.5)
        weight = self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm
        return weight + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            prob = np.random.rand(*x.shape)
            return x * Tensor(np.where(prob < 1 - self.p, 1., 0.) / np.where(prob < 1 - self.p, 1 - self.p, 1.), dtype=x.dtype)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, i, o, k, stride=1, bias=True, padding=None, device=None, dtype="float32"):
        super().__init__()
        if isinstance(k, tuple):
            k = k[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = i
        self.out_channels = o
        self.kernel_size = k
        self.stride = stride
        self.padding = padding

        ### BEGIN YOUR SOLUTION
        weight_init = init.kaiming_uniform(i* k**2, o* k**2, shape=(k, k, i, o),
            dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_init)

        if bias:
            x = 1.0 / ((i* k**2) ** 0.5)
            self.bias = Parameter(init.rand(o, low=-x, high=x,
                dtype=dtype, device=device, requires_grad=True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        xt = x.transpose(axes=(1,2)).transpose(axes=(2,3))

        if self.padding is None:
            padding = self.kernel_size//2
        else:
            padding = self.padding

        out = ops.conv(xt, self.weight, padding=padding, stride=self.stride)
        out = out.transpose(axes=(2,3)).transpose(axes=(1,2))

        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_channels, 1, 1))
            out += bias.broadcast_to(out.shape)

        return out
        ### END YOUR SOLUTION


class Deconv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, i, o, k, stride=1, bias=True, padding=None, device=None, dtype="float32"):
        super().__init__()
        if isinstance(k, tuple):
            k = k[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = i
        self.out_channels = o
        self.kernel_size = k
        self.stride = stride
        self.padding = padding

        weight_init = init.kaiming_uniform(i* k**2, o* k**2, shape=(k, k, i, o),
            dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_init)

        if bias:
            x = 1.0 / ((i* k**2) ** 0.5)
            self.bias = Parameter(init.rand(o, low=-x, high=x,
                dtype=dtype, device=device, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        xt = x.transpose(axes=(1,2)).transpose(axes=(2,3))

        if self.padding is None:
            padding = self.kernel_size//2
        else:
            padding = self.padding

        out = ops.deconv(xt, self.weight, padding=padding, stride=self.stride)
        out = out.transpose(axes=(2,3)).transpose(axes=(1,2))

        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_channels, 1, 1))
            out += bias.broadcast_to(out.shape)

        return out


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        distr = 1 / hidden_size ** 0.5

        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-distr, high=distr), dtype=dtype, device=device)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-distr, high=distr), dtype=dtype, device=device)
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-distr, high=distr), dtype=dtype, device=device)
            self.bias_hh = Parameter(init.rand(hidden_size, low=-distr, high=distr), dtype=dtype, device=device)

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape

        if h is None:
            h = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)

        out = X @ self.W_ih + h @ self.W_hh

        if self.bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))

        if self.nonlinearity == 'tanh':
            return ops.tanh(out)
        else:
            return ops.relu(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
            dtype=dtype, device=device)]
        self.rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
            dtype=dtype, device=device) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape

        if h0 is None:
            hiddens = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
        else:
            hiddens = list(ops.split(h0, 0))

        outputs = list(ops.split(X, 0))

        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]
            for i in range(seq_len):
                x = rnn_cell(outputs[i], hiddens[layer])
                hiddens[layer] = x
                outputs[i] = x

        return ops.stack(outputs, 0), ops.stack(hiddens, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        distr = 1 / hidden_size ** 0.5

        self.W_ih = Parameter(init.rand( input_size, 4*hidden_size, low=-distr, high=distr), device=device)
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-distr, high=distr), device=device)

        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-distr, high=distr), device=device)
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-distr, high=distr), device=device)
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)
            c0 = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)
        else:
            h0, c0 = h

        out = X @ self.W_ih + h0 @ self.W_hh

        if self.bias:
            out += self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)

        _sigmoid = Sigmoid()
        i, f, g, o = ops.split(out, 1, chunks=4)
        i, f, g, o = _sigmoid(i), _sigmoid(f), ops.tanh(g), _sigmoid(o)

        c = f * c0 + i * g
        h = o * ops.tanh(c)

        return h, c

        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, dtype=dtype, device=device)]

        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device)
            for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape

        if h is None:
            hiddens_h = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
            hiddens_c = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
        else:
            h0, c0 = h
            hiddens_h = list(ops.split(h0, 0))
            hiddens_c = list(ops.split(c0, 0))

        outputs = list(ops.split(X, 0))

        for layer in range(self.num_layers):
            lstm_cell = self.lstm_cells[layer]
            for i in range(seq_len):
                h, c = lstm_cell(outputs[i], (hiddens_h[layer], hiddens_c[layer]))
                hiddens_h[layer] = h
                hiddens_c[layer] = c
                outputs[i] = h

        return ops.stack(outputs, 0), (ops.stack(hiddens_h, 0), ops.stack(hiddens_c, 0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device))
        self.eye = np.eye(num_embeddings, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        y = Tensor(self.eye[x.numpy().astype("int")], device=x.device)
        y.requires_grad = False

        return y @ self.weight
        ### END YOUR SOLUTION
