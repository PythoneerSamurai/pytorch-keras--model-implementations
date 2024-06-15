import torch.nn as nn
from torchinfo import summary

class AlexNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.mp = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.fc1 = nn.Linear(in_features=256*3*3, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        c1 = self.c1(x)
        c1 = nn.functional.relu(c1)
        bn = self.bn1(c1)
        mp = self.mp(bn)

        c2 = self.c2(mp)
        c2 = nn.functional.relu(c2)
        bn = self.bn2(c2)
        mp = self.mp(bn)

        c3 = self.c3(mp)
        c3 = nn.functional.relu(c3)
        c4 = self.c4(c3)
        c4 = nn.functional.relu(c4)
        c5 = self.c5(c4)
        c5 = nn.functional.relu(c5)
        mp = self.mp(c5)

        flatten = mp.view(-1, 256*3*3)

        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)

        output = self.fc3(fc2)

        return nn.functional.softmax(output)

MODEL = AlexNet()
print(summary(MODEL, (1, 3, 227, 227)))
