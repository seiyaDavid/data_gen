# src/models/vae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from src.utils import setup_logging
from src.config import config

logger = setup_logging(config.LOG_FILE_PATH)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

    def forward(self, x):
        logger.debug(f"Encoder input x shape: {x.shape}")
        hidden1 = self.relu1(self.linear1(x))
        hidden2 = self.relu2(self.linear2(hidden1))
        mu = self.mu_layer(hidden2)
        logvar = self.logvar_layer(hidden2)
        logger.debug(
            f"Encoder output mu shape: {mu.shape}, logvar shape: {logvar.shape}"
        )
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        logger.debug(
            f"Decoder initialized with latent_dim: {latent_dim}, output_dim: {output_dim}"
        )
        self.linear1 = nn.Linear(latent_dim, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, output_dim)
        self.relu3 = nn.ReLU()  # Ensure ReLU is the final activation

    def forward(self, z):
        logger.debug(f"Decoder input z shape: {z.shape}")
        hidden1 = self.relu1(self.linear1(z))
        hidden2 = self.relu2(self.linear2(hidden1))

        # Debug prints around self.linear3
        logger.debug(f"Decoder input to linear3 (hidden2) shape: {hidden2.shape}")
        logger.debug(f"Decoder linear3 weight shape: {self.linear3.weight.shape}")
        reconstructed_x = self.relu3(self.linear3(hidden2))  # Use ReLU activation
        logger.debug(f"Decoder output (reconstructed_x) shape: {reconstructed_x.shape}")

        return reconstructed_x


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        logger.debug(
            f"VAE initialized with input_dim: {input_dim}, latent_dim: {latent_dim}"
        )
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        logger.debug(f"VAE input x shape: {x.shape}")
        mu, logvar = self.encoder(x)
        logger.debug(
            f"VAE encoder output mu shape: {mu.shape}, logvar shape: {logvar.shape}"
        )
        z = self.reparameterize(mu, logvar)
        logger.debug(f"VAE reparameterized z shape: {z.shape}")
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar


def loss_function_vae(reconstructed_x, x, mu, logvar):
    """VAE loss function (reconstruction loss + KL divergence)."""
    reconstruction_loss = nn.MSELoss(reduction="sum")(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def train_vae(train_loader, feature_dim, latent_dim, epochs, learning_rate):
    """Trains a VAE model."""
    model = VAE(feature_dim, latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting VAE training...")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0].to(device)
            else:
                data = data.to(device)

            optimizer.zero_grad()
            reconstructed_data, mu, logvar = model(data)
            loss = loss_function_vae(reconstructed_data, data, mu, logvar)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logger.info(
                    f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(train_loader)}], Loss VAE: {loss.item():.4f}"
                )

    logger.info("VAE training finished.")
    return model
