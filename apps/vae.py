import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import needle.init as init
from needle import ops
from needle.data import MNISTDataset, DataLoader
from PIL import Image
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, device=ndl.cpu_numpy(), dtype='float32'):
        super().__init__()
        self.img2hid = nn.Linear(input_dim, h_dim, device=device, dtype=dtype)
        self.hid2mu = nn.Linear(h_dim, z_dim, device=device, dtype=dtype)
        self.hid2sigma = nn.Linear(h_dim, z_dim, device=device, dtype=dtype)

        self.z2hid = nn.Linear(z_dim, h_dim, device=device, dtype=dtype)
        self.hid2img = nn.Linear(h_dim, input_dim, device=device, dtype=dtype)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.img2hid(x))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z2hid(z))
        return self.sigmoid(self.hid2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = init.rand(*sigma.shape, device=x.device)
        z_new = mu + sigma * epsilon
        x_rec = self.decode(z_new)
        return x_rec, mu, sigma


def train_vae():
    device = ndl.cuda()
    input_dim = 784
    h_dim = 100
    z_dim = 10
    num_epochs = 3
    batch_size = 64
    lr = 3e-4

    dataset = MNISTDataset("./data/train-images-idx3-ubyte.gz",
                             "./data/train-labels-idx1-ubyte.gz")

    train_loader = DataLoader(dataset, batch_size)
    model = VAE(input_dim, h_dim, z_dim, device=device)
    optimizer = ndl.optim.Adam(model.parameters(), lr=lr, device=device)
    loss_fn = nn.BinaryCrossEntropy()

    gen_images = []

    for epoch in range(1, num_epochs + 1):
        for iter, (x, t) in enumerate(train_loader):
            x = ndl.Tensor(x.reshape((x.shape[0], input_dim)), device=device)
            x_rec, mu, sigma = model(x)

            rec_loss = loss_fn(x_rec, x)
            # Computing Kullback-Leibler Divergence
            kl_loss = (mu ** 2 + ops.exp(sigma) - sigma - 1) * 0.5
            kl_loss = ops.summation(kl_loss) / x.shape[0]

            loss = rec_loss + kl_loss
            optimizer.reset_grad()
            loss.backward()
            optimizer.step()

            if iter % 300 == 0:
                print('epoch: {}, iter: {}, loss: {}'.format(epoch, iter, loss.data))
                np_img = x_rec.reshape((-1, 1, 28, 28)).data.numpy()
                np_img = np_img[:1,...].reshape((28, 28))
                im = Image.fromarray(np.uint8(np_img * 255))
                gen_images.append(im)
                im.save(f'vae_{epoch}_{iter}_epoch.jpg')
