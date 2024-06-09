import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding='same')
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')

        self.mp = nn.MaxPool2d(kernel_size=(2, 2))

        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same')

        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')
        self.c7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')
        self.c8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')

        self.c9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')

        self.c13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')

        self.fc1 = nn.Linear(in_features=25088, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        output = self.c1(x)
        output = nn.functional.relu(output)
        output = self.c2(output)
        output = nn.functional.relu(output)
        output = self.mp(output)
        output = self.c3(output)
        output = nn.functional.relu(output)
        output = self.c4(output)
        output = nn.functional.relu(output)
        output = self.mp(output)
        output = self.c5(output)
        output = nn.functional.relu(output)
        output = self.c6(output)
        output = nn.functional.relu(output)
        output = self.c7(output)
        output = nn.functional.relu(output)
        output = self.c8(output)
        output = self.mp(output)
        output = self.c9(output)
        output = nn.functional.relu(output)
        output = self.c10(output)
        output = nn.functional.relu(output)
        output = self.c11(output)
        output = nn.functional.relu(output)
        output = self.c12(output)
        output = self.mp(output)
        output = self.c13(output)
        output = nn.functional.relu(output)
        output = self.c14(output)
        output = nn.functional.relu(output)
        output = self.c15(output)
        output = nn.functional.relu(output)
        output = self.c16(output)
        output = self.mp(output)
        output = output.view(-1, 25088)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

from torchinfo import summary
MODEL = VGG19()
print(summary(MODEL, (3, 224, 224)))
