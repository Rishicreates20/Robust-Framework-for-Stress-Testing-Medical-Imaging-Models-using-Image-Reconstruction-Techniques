import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Path to dataset
dataset_zip_path = r"C:\Users\rishi\OneDrive\Desktop\mini project\archive (1).zip"
extracted_path = "./ECG_Image_data"

# Extract dataset if not already done
if not os.path.exists(extracted_path):
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

# Check if the extraction worked correctly
if not os.path.exists(extracted_path):
    print("Dataset folder not found. Please check the path.")
    exit(1)

# Debug: List files in the dataset folder recursively
for root, dirs, files in os.walk(extracted_path):
    print("Directory:", root)
    for file in files:
        print("File:", file)

# Ensure the dataset folder contains .png files
image_files = [os.path.join(root, file) for root, dirs, files in os.walk(extracted_path) for file in files if file.endswith('.png')]
if not image_files:
    print("No .png files found in the dataset folder. Please check your dataset.")
    exit(1)

print(f"Found {len(image_files)} .png files.")

# Hyperparameters
latent_dim = 100
img_size = 64
channels = 3
batch_size = 128
epochs = 100
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * channels, [0.5] * channels)
])

# Custom Dataset Loader
class CustomImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

dataset = CustomImageDataset(image_files=image_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),       # Downsample
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),       # Downsample
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),         # Output layer
            nn.Sigmoid()
        )

    def forward(self, img):
        output = self.model(img)  # Output shape: [batch_size, 1, 1, 1]
        return output.view(img.size(0), -1)  # Flatten to [batch_size, 1]


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
real_imgs = torch.randn(128, channels, img_size, img_size).to(device)  # Simulate a batch of real images
output = discriminator(real_imgs)
print(f"Discriminator output shape: {output.shape}")  # Should print: [128, 1]


# Optimizers and Loss
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Log Progress
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Save generated images
    os.makedirs("images", exist_ok=True)
    save_image(fake_imgs[:25], f"images/{epoch}.png", nrow=5, normalize=True)

# Visualize Results
def show_generated_images(epoch):
    img = make_grid(fake_imgs[:25], nrow=5, normalize=True).permute(1, 2, 0).cpu()
    plt.imshow(img)
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.axis("off")
    plt.show()

show_generated_images(epochs - 1)






