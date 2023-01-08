"""Operatpr table."""
# Global operator table.
import copy
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * node.inputs[0] ** (self.scalar-1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        gx0 = out_grad / rhs
        gx1 = out_grad * (-lhs / rhs**2)
        return gx0, gx1
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (a / self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return a.swap_axes(self.axes[0], self.axes[1])
        else:
        	n = len(a.shape)
        	return a.swap_axes(n-1, n-2)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes),
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return reshape(out_grad, lhs.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.new_shape = None

    def compute(self, a):
        n = len(a.shape)
        if n == len(self.shape):
            self.new_shape = a.shape
        else:
            new_shape = []
            k = 0
            for i in range(len(self.shape)):
                if n == 0 or k == n:
                    new_shape.insert(0, 1)
                else:
                    new_shape.append(a.shape[k])
                    k += 1

            self.new_shape = tuple(new_shape)
            a = a.reshape(self.new_shape)

        return array_api.broadcast_to(a, self.shape)
        # return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        diff = []
        for axe in range(len(out_grad.shape)):
            if out_grad.shape[axe] != node.inputs[0].shape[axe]:
                diff.append(axe)

        out_grad = summation(out_grad, axes=tuple(diff))
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes, keepdims=self.keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if out_grad.shape == node.inputs[0].shape:
            return out_grad

        shape = list(node.inputs[0].shape)
        if self.axes is None:
            self.axes = list(range(len(node.inputs[0].shape)))

        if isinstance(self.axes, int):
            self.axes = [self.axes]
        for axe in self.axes:
            shape[axe] = 1

        out_grad = reshape(out_grad, shape)
        out_grad = broadcast_to(out_grad, node.inputs[0].shape)
        return out_grad
        ### END YOUR SOLUTION


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        assert a.shape[-1] == b.shape[-2], "MatMul: Sizes not matched %s @ %s" % (a.shape, b.shape)

        # 2D @ 2D
        if a.ndim == 2 and b.ndim == 2: return a @ b

        # 2D @ nD
        if a.ndim == 2:
            if b.ndim == 3:
                c = b.permute((1,2,0)).reshape((b.shape[1], b.shape[2]*b.shape[0]))
                return (a @ c).reshape((a.shape[0], b.shape[2], b.shape[0])).permute((2,0,1))

            assert b.ndim == 4, "MatMul: Only support 2D @ 3,4D MatMul"
            c = b.permute((2,3,0,1)).reshape((b.shape[2], b.shape[3]*b.shape[0]*b.shape[1]))
            return (a @ c).reshape((a.shape[0], b.shape[3], b.shape[0], b.shape[1])).permute((3,0,1,2))

        # nD @ 2D
        if b.ndim == 2:
            assert a.ndim >= 3, "MatMul: a.ndim must >= 3 for nD @ 2D"
            c = a.reshape((-1, a.shape[-1]))
            shape = list(a.shape)
            shape[-1] = b.shape[1]
            return (c @ b).reshape(shape)

        # 3D @ 3D
        if a.ndim == 3 and b.ndim == 3:
            assert a.shape[0] == b.shape[0], "MatMul: Batch need to be same size"
            c = NDArray.make((a.shape[0], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                _a = a[i,:,:].reshape((a.shape[-2], a.shape[-1]))
                _b = b[i,:,:].reshape((b.shape[-2], b.shape[-1]))
                c[i,:,:] = (_a @ _b).reshape((1, a.shape[-2], b.shape[-1]))
            return c

        # >>> (3, 2, 1) (3, 3, 1, 2)
        if b.ndim == 4 and a.ndim == 3 and b.shape[1] == a.shape[0]:
            c = NDArray.make((b.shape[0], a.shape[-3], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                _a = a[i,:,:].reshape((a.shape[-2], a.shape[-1]))
                for j in range(b.shape[0]):
                    _b = b[j,i,:,:].reshape((b.shape[-2], b.shape[-1]))
                    c[j,i,:,:] = (_a @ _b).reshape((1, 1, a.shape[-2], b.shape[-1]))
            return c

        # >>> (3, 3, 2, 2) (3, 3, 2, 1)
        if b.ndim == 4 and a.ndim == 4 and b.shape[0] == a.shape[0] and b.shape[1] == a.shape[1]:
            c = NDArray.make((a.shape[-4], a.shape[-3], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                for j in range(b.shape[1]):
                    _a = a[i,j,:,:].reshape((a.shape[-2], a.shape[-1]))
                    _b = b[i,j,:,:].reshape((b.shape[-2], b.shape[-1]))
                    c[i,j,:,:] = (_a @ _b).reshape((1, 1, a.shape[-2], b.shape[-1]))
            return c

        assert False, "MatMul: Only support 2D @ 2D, nD @ 2D, 2D @ 3:4D, 3D @ 3D and selected 4D @ 4D"
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        res1 = out_grad @ rhs.transpose()
        res2 = lhs.transpose() @ out_grad

        if res1.shape != lhs.shape:
            diff = len(res1.shape) - len(lhs.shape)
            res1 = summation(res1, axes=tuple(range(0, diff)))
        if res2.shape != rhs.shape:
            diff = len(res2.shape) - len(rhs.shape)
            res2 = summation(res2, axes=tuple(range(0, diff)))

        return res1, res2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        t = init.ones(*node.inputs[0].shape, device=node.inputs[0].device)
        return t / node.inputs[0] * out_grad
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Clip(TensorOp):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def compute(self, x):
        y = array_api.clip(x, self.x_min, self.x_max)
        return y

    def gradient(self, out_grad, node):
        x, = node.inputs
        mask = Tensor((x.data.numpy() >= self.x_min) * (x.data.numpy() <= self.x_max),
                        dtype=out_grad.dtype, device=out_grad.device)
        gx = out_grad * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return exp(node.inputs[0]) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0.)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor((node.inputs[0].numpy() > 0) * 1.0,
                        dtype=out_grad.dtype, device=out_grad.device) * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max_orig = Z.max(axis=self.axes)
        Z_max = Z_max_orig

        shape = list(Z.shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                self.axes = [self.axes]
            axes = self.axes
        else:
            axes = list(Z.shape)

        for axe in axes:
            shape[axe] = 1

        Z_max = array_api.reshape(Z_max_orig, shape)
        Z_max = array_api.broadcast_to(Z_max, Z.shape)

        return array_api.log(array_api.exp(Z - Z_max).sum(axis=self.axes)) + Z_max_orig
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]

        shape = list(x.shape)
        if self.axes is None:
            self.axes = list(range(len(x.shape)))

        for axe in self.axes:
            shape[axe] = 1

        out_grad = reshape(out_grad, shape)
        out_grad = broadcast_to(out_grad, x.shape)

        node = reshape(node, shape)
        node = broadcast_to(node, x.shape)

        return exp(x - node) * out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (1.0 - tanh(node.inputs[0]) ** 2) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        shape.insert(self.axis, len(args))
        idxs = [slice(0, shape[i], 1) for i in range(len(shape))]
        out = NDArray.make(shape, device = args[0].device)
        for i, tensor in enumerate(args):
            assert args[0].shape == tensor.shape, "stacked tensors must be same shape"
            idxs[self.axis] = slice(i, i + 1, 1)
            out[tuple(idxs)] = tensor
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int, chunks=None):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis
        self.chunks = chunks

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = list(A.shape)
        idxs = [slice(0,shape[i],1) for i in range(len(shape))]

        if self.chunks is None:
            del shape[self.axis]
            chunks = A.shape[self.axis]
            offset = 1
        else:
            chunks = self.chunks
            offset = A.shape[self.axis] // chunks
            shape[self.axis] = offset

        out = []
        for i in range(chunks):
            start = i * offset
            idxs[self.axis] = slice(start, start + offset, 1)
            a = A.__getitem__(tuple(idxs))
            out.append(a.reshape(shape))

        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        assert isinstance(out_grad, TensorTuple)
        return Stack(self.axis)(out_grad).reshape(input_shape),
        ### END YOUR SOLUTION


def split(a, axis, chunks=None):
    return Split(axis, chunks=chunks)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for i in range(len(a.shape)):
            if i in self.axes:
                new_shape[i] = new_shape[i] * (self.dilation + 1)
                idxs[i] = slice(0, new_shape[i], (self.dilation + 1))
        out = a.device.full(new_shape, 0.)
        out[tuple(idxs)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for i in range(len(a.shape)):
            if i in self.axes:
                idxs[i] = slice(0, a.shape[i], (self.dilation + 1))
        return a[tuple(idxs)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if self.padding > 0:
            axes = []
            for i in range(len(A.shape)):
                if i in [1, 2]: # H, W
                    axes.append((self.padding, self.padding))
                else:
                    axes.append((0, 0))
            A = A.pad(axes)

        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        A_2 = A.as_strided((N, H-K+1, W-K+1, K, K, C_in), (Ns, Hs, Ws, Hs, Ws, Cs))
        A_2 = A_2.compact().reshape((-1, inner_dim))
        B_2 = B.compact().reshape((-1, C_out))

        out = A_2 @ B_2
        out = out.reshape((N,H-K+1,W-K+1,C_out))

        if self.stride > 1:
            idxs = [slice(0, out.shape[i], 1) for i in range(len(out.shape))]
            for i in range(len(out.shape)):
                if i in (1,2):
                    idxs[i] = slice(0, out.shape[i], self.stride)
            out = out[tuple(idxs)].compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs

        if self.stride > 1:
            out_grad = dilate(out_grad, (1,2), self.stride - 1) # NWHC

        pad = X.shape[1] - out_grad.shape[1] + self.padding
        X_grad = conv(out_grad, flip(W, axes=(0,1)).transpose(), padding=pad)

        out_grad = out_grad.transpose(axes=(0,2)).transpose(axes=(0,1))
        W_grad = conv(X.transpose(axes=(0,3)), out_grad, padding=self.padding)
        W_grad = W_grad.transpose(axes=(0,2)).transpose(axes=(0,1))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


class Deconv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        if self.stride > 1:
            A = A.dilate((1,2), self.stride - 1)

        self.stride = 1
        self.padding = B.shape[0] - self.padding - 1

        if self.padding > 0:
            axes = []
            for i in range(len(A.shape)):
                if i in [1, 2]: # H, W
                    axes.append((self.padding, self.padding))
                else:
                    axes.append((0, 0))
            A = A.pad(axes)

        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * C_in
        A_2 = A.as_strided((N, H-K+1, W-K+1, K, K, C_in), (Ns, Hs, Ws, Hs, Ws, Cs))
        A_2 = A_2.compact().reshape((-1, inner_dim))

        out = A_2 @ B.compact().reshape((-1, C_out)).flip(axes=(0,1))
        out = out.reshape((N, H-K+1, W-K+1, C_out))

        return out

    def gradient(self, out_grad, node):
        X, W = node.inputs

        if self.stride > 1:
            out_grad = undilate(out_grad, (1,2), self.stride - 1) # NWHC

        print(W.shape)
        print(out_grad.shape)
        X_grad = conv(out_grad, W)

        out_grad = out_grad.transpose(axes=(0,2)).transpose(axes=(0,1))

        W_grad = conv(out_grad, X.transpose(axes=(0,3)), padding=self.padding)
        W_grad = W_grad.transpose(axes=(0,2)).transpose(axes=(0,1))

        return X_grad, W_grad

def deconv(a, b, stride=1, padding=1):
    return Deconv(stride, padding)(a, b)