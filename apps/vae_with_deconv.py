import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import needle.init as init
from needle import ops
from needle.data import MNISTDataset, DataLoader

import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt

# https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/examples/vae.py


class Encoder(nn.Module):
    def __init__(self, latent_size, device=ndl.cpu_numpy(), dtype='float32'):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv(1, 32, 3, stride=1, device=device, dtype=dtype)
        self.conv2 = nn.Conv(32, 64, 3, stride=2, device=device, dtype=dtype)
        self.conv3 = nn.Conv(64, 64, 3, stride=1, device=device, dtype=dtype)
        self.conv4 = nn.Conv(64, 64, 3, stride=1, device=device, dtype=dtype)
        self.linear1 = nn.Linear(12544, 32, device=device, dtype=dtype)
        self.linear2 = nn.Linear(32, latent_size, device=device, dtype=dtype)
        self.linear3 = nn.Linear(32, latent_size, device=device, dtype=dtype)

    def forward(self, x):
        x = ops.relu(self.conv1(x))
        x = ops.relu(self.conv2(x))
        x = ops.relu(self.conv3(x))
        x = ops.relu(self.conv4(x))
        x = nn.Flatten()(x)
        x = ops.relu(self.linear1(x))
        z_mean = self.linear2(x)
        z_log_var = self.linear3(x)
        return z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        epsilon = init.rand(batch_size, self.latent_size)
        return z_mean + ops.exp(z_log_var) * epsilon


class Decoder(nn.Module):
    def __init__(self, device=ndl.cpu_numpy(), dtype='float32'):
        super().__init__()
        self.to_shape = (64, 14, 14)
        self.linear = nn.Linear(2, np.prod(self.to_shape), device=device)
        self.deconv = nn.Deconv(64, 32, 3, stride=2, padding=1, device=device)
        self.conv = nn.Conv(32, 1, 3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = ops.relu(self.linear(x))
        x = ops.reshape(x, (-1,) + self.to_shape)
        x = ops.relu(self.deconv(x))
        x = self.conv(x)
        print(x.shape)
        x = nn.Sigmoid()(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder()

    def forward(self, x, C=1.0, k=1):
        """Call loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            x (Variable or ndarray): Input variable.
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        z_mean, z_log_var = self.encoder(x)
        rec_loss = 0
        for l in range(k):
            z = self.encoder.sampling(z_mean, z_log_var)
            y = self.decoder(z)

            loss_fn = nn.BinaryCrossEntropy()
            flatten = nn.Flatten()

            rec_loss += loss_fn(flatten(y), flatten(x)) / k

        kl_loss = C * (z_mean ** 2 + ops.exp(z_log_var) - z_log_var - 1) * 0.5
        kl_loss = ops.summation(kl_loss) / x.shape[0]
        print(rec_loss)
        print(kl_loss)
        return rec_loss + kl_loss

    def show_digits(epoch=0):
        """Display a 2D manifold of the digits"""
        n = 15  # 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        # grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])

                with dezero.no_grad():
                    x_decoded = vae.decoder(z_sample)

                digit = x_decoded.data.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        # plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.imshow(figure, cmap='Greys_r')
        # plt.show()
        # plt.savefig('vae_{}.png'.format(epoch))


def training():
    max_epoch = 10
    batch_size = 16
    latent_size = 2

    vae = VAE(latent_size)
    optimizer = ndl.optim.Adam(vae.parameters())

    train_set = MNISTDataset("./data/train-images-idx3-ubyte.gz",
                             "./data/train-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size)

    for epoch in range(max_epoch):
        avg_loss = 0
        cnt = 0

        for x, t in train_loader:
            cnt += 1
            x = x.reshape((-1, 1, 28, 28))

            loss = vae(x)
            optimizer.reset_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.data.numpy()
            interval = 10
            max_iter = 3750

            if cnt % interval == 0:
                epoch_detail = epoch + cnt / max_iter
                print('epoch: {:.2f}, loss: {:.4f}'.format(epoch_detail,
                                                        float(avg_loss/cnt)))

        show_digits(epoch)