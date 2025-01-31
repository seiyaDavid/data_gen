# src/models/gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import setup_logging
from src.config import config

logger = setup_logging(config.LOG_FILE_PATH)


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Increase neurons here
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),  # Increase neurons here
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),  # Add a layer here
            nn.LeakyReLU(0.2),  # Add activation function
            nn.BatchNorm1d(1024),  # Add batch normalization
            nn.Linear(1024, 512),  # Add a layer here
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),  # Increase neurons here
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        return self.model(z)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output probability (real or fake)
        )

    def forward(self, x):
        return self.model(x)


def train_gan(train_loader, feature_dim, latent_dim, epochs, learning_rate):
    """Trains a GAN model."""
    generator = Generator(latent_dim, feature_dim)
    discriminator = Discriminator(feature_dim)

    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss

    logger.info("Starting GAN training...")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            # --- Handle potential list/tuple output from DataLoader ---
            if isinstance(data, list) or isinstance(data, tuple):
                real_data = data[0].to(device)  # Access tensor from list/tuple
            else:
                real_data = data.to(device)  # Assume it's already a tensor
            # --- End of DataLoader output handling ---

            batch_size = real_data.size(0)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            # Real data loss
            label_real = torch.ones(batch_size, 1).to(device)
            output_real = discriminator(real_data)
            loss_D_real = criterion(output_real, label_real)

            # Fake data loss
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(noise)
            label_fake = torch.zeros(batch_size, 1).to(device)
            output_fake = discriminator(
                fake_data.detach()
            )  # Detach to not backpropagate through Generator during D training
            loss_D_fake = criterion(output_fake, label_fake)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            label_real_generator = torch.ones(batch_size, 1).to(
                device
            )  # Generator wants Discriminator to think fakes are real
            output_generator = discriminator(
                fake_data
            )  # Re-evaluate discriminator on the *updated* fake data
            loss_G = criterion(output_generator, label_real_generator)
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                logger.info(
                    f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(train_loader)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
                )

    logger.info("GAN training finished.")
    return generator, discriminator
