import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from tqdm import tqdm

class Critic(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            self._block(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            self._block(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=feature_dim * 8, out_channels=1, kernel_size=4, stride=2, padding=0),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )


    def forward(self, x):
        return self.critic(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, feature_g):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise, feature_g * 16, 4, 1, 0),
            self._block(feature_g * 16, feature_g * 8, 4, 2, 1),
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                feature_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transform.Compose([
    transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transform.ToTensor(),
    transform.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
    )
])

DATA = torchvision.datasets.ImageFolder(root="/kaggle/input/person-face-dataset-thispersondoesnotexist", transform=transforms)
loader = DataLoader(dataset=DATA, shuffle=True, batch_size=BATCH_SIZE)

gen = nn.DataParallel(Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device))
critic = nn.DataParallel(Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device))

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

for epoch in range(NUM_EPOCHS):
    for batch_id, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        current_batch_size = data.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(current_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, data, fake, "cuda")
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake))+LAMBDA_GP *gp)
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        with torch.no_grad():
            save_plot(critic(fake).cpu(), epoch)

        if batch_id % 100 == 0 and batch_id > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_id}/{len(loader)} \
                              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
