import os
import scanpy as sc

import torch
import torch.nn.functional as F

from loss import ZINB

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Model(torch.nn.Module):

    def __init__(self, in_size, hidden_size, z_size):
        super(Model, self).__init__()

        self.encoder_1 = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU()
        )

        self.encoder_2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, z_size),
            torch.nn.BatchNorm1d(z_size),
            torch.nn.ReLU()
        )

        self.decoder_1 = torch.nn.Sequential(
            torch.nn.Linear(z_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU()
        )

        self.pi = torch.nn.Linear(hidden_size, in_size)
        self.disp = torch.nn.Linear(hidden_size, in_size)
        self.mean = torch.nn.Linear(hidden_size, in_size)

        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, x):
        z = self.encoder_2(self.encoder_1(x))

        x = self.decoder_1(z)

        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))

        return [pi, disp, mean]


# %%
adata = sc.read_h5ad('.h5ad')
# clusters = adata.obs.cluster.values

model = Model(adata.shape[1], 256, 64).cuda()
optimizer = torch.optim.Adam(model.parameters())

criterion = ZINB()

model.train()

for epoch in range(200):
    optimizer.zero_grad()

    data = torch.tensor(adata.X.A).cuda()
    label = torch.tensor(adata.X.A).cuda()
    pi, disp, mean = model(data)

    loss = criterion(pi, disp, label, mean) + 0.1 * F.mse_loss(mean, label)

    loss.backward()
    optimizer.step()
