import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.mp = nn.MaxPool2d(kernel_size=(2, 2))
        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding='same')
        self.dp1 = nn.Dropout(0.1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')

        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same')

        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same')
        self.dp2 = nn.Dropout(0.2)
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')

        self.c7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')

        self.c9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding='same')
        self.dp3 = nn.Dropout(0.3)
        self.c10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same')

        self.ct1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=(2, 2))
        self.c11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')
        self.c12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same')

        self.ct2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        self.c13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')
        self.c14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same')

        self.ct3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.c15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same')
        self.c16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same')

        self.ct4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.c17 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')
        self.c18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')

        self.c19 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1))

    def forward(self, x):
        # Encoder Path
        c1 = self.c1(x)
        dp1 = self.dp1(nn.functional.relu(c1))
        c2 = self.c2(dp1)
        c3 = self.c3(self.mp(c2))
        dp1 = self.dp1(nn.functional.relu(c3))
        c4 = self.c4(dp1)
        c5 = self.c5(self.mp(c4))
        dp2 = self.dp2(nn.functional.relu(c5))
        c6 = self.c6(dp2)
        c7 = self.c7(self.mp(c6))
        dp2 = self.dp2(nn.functional.relu(c7))
        c8 = self.c8(dp2)
        c9 = self.c9(self.mp(c8))
        dp3 = self.dp3(nn.functional.relu(c9))
        c10 = self.c10(dp3)
        # Decoder Path
        ct1 = self.ct1(c10)
        ct1 = torch.concatenate((c8, ct1))
        c11 = self.c11(ct1)
        dp3 = self.dp3(nn.functional.relu(c11))
        c12 = self.c12(dp3)
        ct2 = self.ct2(nn.functional.relu(c12))
        ct2 = torch.concatenate((c6, ct2))
        c13 = self.c13(ct2)
        dp2 = self.dp3(nn.functional.relu(c13))
        c14 = self.c14(dp2)
        ct3 = self.ct3(nn.functional.relu(c14))
        ct3 = torch.concatenate((c4, ct3))
        c15 = self.c15(ct3)
        dp2 = self.dp2(nn.functional.relu(c15))
        c16 = self.c16(dp2)
        ct4 = self.ct4(nn.functional.relu(c16))
        ct4 = torch.concatenate((c2, ct4))
        c17 = self.c17(ct4)
        dp1 = self.dp1(nn.functional.relu(c17))
        c18 = self.c18(dp1)
        c19 = self.c19(nn.functional.relu(c18))

        return nn.functional.sigmoid(c19)


MODEL = UNET()

from torchinfo import summary

print(summary(MODEL, (1, 3, 128, 128)))
