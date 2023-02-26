import sys
sys.path.append('./python')

import numpy as np
import torch
from torch import nn
import needle as ndl

np.random.seed(0)
Z_shape = (1, 12, 12, 16)
Z_h_shape = (1, 6, 6, 16)
W_shape = (3, 3, 16, 16)
device = ndl.cpu_numpy()

# _Z = np.random.randn(*Z_shape)

# input_Z = ndl.Tensor(_Z, device=ndl.cpu_numpy())
# input = torch.Tensor(_Z)

# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

# downsample_ndl = ndl.nn.Conv(16, 16, 3, stride=2, padding=1)
# upsample_ndl = ndl.nn.Deconv(16, 16, 3, stride=2, padding=1)

# h = downsample(input)
# print(h.size())
# output = upsample(h, output_size=input.size())
# print(output.size())

# h_ndl = downsample_ndl(input_Z)
# print(h_ndl.shape)
# output_ndl = upsample_ndl(h_ndl)
# print(output_ndl.shape)

# print(np.linalg.norm(h.detach().numpy() - h_ndl.data.numpy()))
# print(np.linalg.norm(output.detach().numpy() - output_ndl.data.numpy()))

padding, stride = 1, 2
backward = True

_Z = np.random.randn(*Z_shape)*5
_Z = _Z.astype(np.float32)
_W = np.random.randn(*W_shape)*5
_W = _W.astype(np.float32)
_Z_h = np.random.randn(*Z_h_shape)*5
_Z_h = _Z.astype(np.float32)

Z = ndl.Tensor(_Z, device=device)
Z_h = ndl.Tensor(_Z_h, device=device)
W = ndl.Tensor(_W, device=device)
y = ndl.conv(Z, W, padding=padding, stride=stride)


y_t = ndl.deconv(Z_h, W, padding=padding, stride=stride)
y_t.backward()
# y2 = y.sum()
# if backward:
#     y2.backward()
Ztch = torch.Tensor(_Z).float()
Ztch.requires_grad=True
Ztch_h = torch.Tensor(_Z_h).float()
Ztch_h.requires_grad=True
Wtch = torch.Tensor(_W).float()
Wtch.requires_grad=True
out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)

out_t = torch.nn.functional.conv_transpose2d(Ztch_h.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride, output_padding=padding).permute(0, 3, 2, 1)
print(out_t[:1,:1,:1,:])
print(y_t.data.numpy()[:1,:1,:1,:])
out2 = out.sum()
if backward:
    out2.backward()
if backward:
    err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
    err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
assert out_t.shape == y_t.shape, "outputs shape mismatch %s, %s" % (out_t.shape, y_t.shape)
err4 = np.linalg.norm(out_t.sum().detach().numpy() - y_t.sum().numpy())
print(err4)
if backward:
    assert err1 < 1e-2, "input grads match"
    assert err2 < 1e-2, "weight grads match"
# assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)