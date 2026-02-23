pip install pandas numpy matplotlib seaborn torch scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


ROLL_NUMBER = 102303221

data = pd.read_csv("/content/data-3.csv", encoding='latin1')

print("DataFrame columns:", data.columns)

x = data["no2"].dropna().values.reshape(-1, 1)


a_r = 0.5 * (ROLL_NUMBER % 7)
b_r = 0.3 * ((ROLL_NUMBER % 5) + 1)

z = x + a_r * np.sin(b_r * x)

print("a_r =", a_r)
print("b_r =", b_r)

scaler = StandardScaler()
z_scaled = scaler.fit_transform(z)

z_tensor = torch.tensor(z_scaled, dtype=torch.float32)

LATENT_DIM = 5
BATCH_SIZE = 128
EPOCHS = 3000
LR = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=LR)
optimizer_D = optim.Adam(D.parameters(), lr=LR)


for epoch in range(EPOCHS):

    idx = np.random.randint(0, z_tensor.size(0), BATCH_SIZE)
    real_samples = z_tensor[idx].to(device)

    real_labels = torch.ones(BATCH_SIZE, 1).to(device)
    fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)

    optimizer_D.zero_grad()

    outputs_real = D(real_samples)
    loss_real = criterion(outputs_real, real_labels)

    noise = torch.randn(BATCH_SIZE, LATENT_DIM).to(device)
    fake_samples = G(noise)

    outputs_fake = D(fake_samples.detach())
    loss_fake = criterion(outputs_fake, fake_labels)

    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    outputs = D(fake_samples)
    loss_G = criterion(outputs, real_labels)

    loss_G.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

print("Training Complete!")

with torch.no_grad():
    noise = torch.randn(10000, LATENT_DIM).to(device)
    generated_samples = G(noise).cpu().numpy()

generated_samples = scaler.inverse_transform(generated_samples)


kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(generated_samples)

z_range = np.linspace(min(z), max(z), 1000).reshape(-1, 1)
log_density = kde.score_samples(z_range)
density = np.exp(log_density)

plt.figure(figsize=(9,5))
plt.hist(z, bins=40, density=True, alpha=0.6, label="Real z (Histogram)")
plt.plot(z_range, density, label="GAN Estimated PDF", linewidth=2)
plt.title("PDF Learned using GAN")
plt.xlabel("z")
plt.ylabel("Density")
plt.legend()
plt.show()
