import torch
import torch.nn as nn
from torchinfo import summary

depth_multiple, width_multiple, ratio = 1.0, 1.25, 1.0


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(),
        )

    def forward(self, data):
        return self.block(data)
    
    
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_blocks = [
            Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
            )
            for _
            in range(2)
        ]
        self.maxpool2d_blocks = [
            nn.MaxPool2d(
                kernel_size=1,
                stride=1,
            )
            for _
            in range(3)
        ]

    def forward(self, data):
        conv_block_one_output = self.conv_blocks[0](data)
        maxpool2d_block_one_output = self.maxpool2d_blocks[0](conv_block_one_output)
        maxpool2d_block_two_output = self.maxpool2d_blocks[1](maxpool2d_block_one_output)
        maxpool2d_block_three_output = self.maxpool2d_blocks[2](maxpool2d_block_two_output)
        concatenated_output = torch.concat(
            [
                conv_block_one_output,
                maxpool2d_block_one_output,
                maxpool2d_block_two_output,
                maxpool2d_block_three_output
            ]
        )
        conv_block_two_output = self.conv_blocks[1](concatenated_output)
        return conv_block_two_output


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.conv_blocks_one = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1
        )
        self.conv_blocks_two = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1
        )

    def forward(self, data):
        conv_block_one_output = self.conv_blocks_one(data)
        conv_block_two_output = self.conv_blocks_two(conv_block_one_output)
        if self.shortcut is True:
            return torch.add(data, conv_block_two_output)
        else:
            return conv_block_two_output


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n, shortcut=True):
        super().__init__()
        self.n = n
        self.shortcut = shortcut
        self.conv_block_one = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0
        )
        self.conv_block_two = Conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0
        )
        self.bottleneck_blocks = [
            BottleNeck(
                in_channels=out_channels,
                out_channels=out_channels,
                shortcut=shortcut
            )
            for _
            in range(n)
        ]

    def forward(self, data):
        conv_block_one_output = self.conv_block_one(data)
        splitted_output = torch.split(conv_block_one_output, 2)
        bottleneck_block_starter_output = self.bottleneck_blocks[0](splitted_output[1])
        bottleneck_blocks_outputs = bottleneck_block_starter_output
        for index, bottleneck_block in enumerate(self.bottleneck_blocks):
            if index == 0:
                pass
            else:
                bottleneck_blocks_outputs = bottleneck_block(bottleneck_blocks_outputs)
        concatenated_output = torch.concat(
            [
                splitted_output[0],
                splitted_output[1],
                bottleneck_block_starter_output,
                bottleneck_blocks_outputs
            ]
        )
        conv_block_two_output = self.conv_block_two(concatenated_output)
        return conv_block_two_output


class Detect(nn.Module):
    def __init__(self, in_channels, out_channels, reg_max=8, nc=1):
        super().__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
        )
        self.bbox_loss_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4*reg_max,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.cls_loss_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=nc,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def bbox_loss_predictor(self, data):
        output = self.conv(data)
        output = self.conv(output)
        bbox_loss = self.bbox_loss_conv2d(output)
        return bbox_loss

    def cls_loss_predictor(self, data):
        output = self.conv(data)
        output = self.conv(output)
        cls_loss = self.cls_loss_conv2d(output)
        return cls_loss

    def forward(self, data):
        bbox_loss = self.bbox_loss_predictor(data)
        cls_loss = self.cls_loss_predictor(data)
        return [bbox_loss, cls_loss]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_one = Conv(
            in_channels=3,
            out_channels=int(64*width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.conv_two = Conv(
            in_channels=int(64 * width_multiple),
            out_channels=int(128 * width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_one = C2f(
            in_channels=int(128 * width_multiple),
            out_channels=int(128 * width_multiple),
            n=int(3*depth_multiple),
            shortcut=True,
        )
        self.conv_three = Conv(
            in_channels=int(128 * width_multiple),
            out_channels=int(256 * width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_two = C2f(
            in_channels=int(256 * width_multiple),
            out_channels=int(256 * width_multiple),
            n=int(6 * depth_multiple),
            shortcut=True,
        )
        self.conv_four = Conv(
            in_channels=int(256 * width_multiple),
            out_channels=int(512 * width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_three = C2f(
            in_channels=int(512 * width_multiple),
            out_channels=int(512 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=True,
        )
        self.conv_five = Conv(
            in_channels=int(512 * width_multiple),
            out_channels=int(512 * width_multiple * ratio),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_four = C2f(
            in_channels=int(512 * width_multiple * ratio),
            out_channels=int(512 * width_multiple * ratio),
            n=int(3 * depth_multiple),
            shortcut=True,
        )
        self.sppf = SPPF(
            in_channels=int(512 * width_multiple * ratio),
            out_channels=int(512 * width_multiple * ratio),
        )

    def forward(self, data):
        conv_one = self.conv_one(data)
        conv_two = self.conv_two(conv_one)
        c2f_one = self.c2f_one(conv_two)
        conv_three = self.conv_three(c2f_one)
        c2f_two = self.c2f_two(conv_three)
        conv_four = self.conv_four(c2f_two)
        c2f_three = self.c2f_three(conv_four)
        conv_five = self.conv_five(c2f_three)
        c2f_four = self.c2f_four(conv_five)
        sppf = self.sppf(c2f_four)
        return c2f_two, c2f_three, sppf


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.upsample_one = nn.Upsample(size=(40, 40))
        self.c2f_one = C2f(
            in_channels=int(512 * width_multiple),
            out_channels=int(256 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.upsample_two = nn.Upsample(size=(80, 80))
        self.c2f_two = C2f(
            in_channels=int(256 * width_multiple),
            out_channels=int(256 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.conv_one = Conv(
            in_channels=int(256 * width_multiple),
            out_channels=int(256 * width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_three = C2f(
            in_channels=int(256 * width_multiple),
            out_channels=int(512 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.conv_two = Conv(
            in_channels=int(512 * width_multiple),
            out_channels=int(512 * width_multiple),
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.c2f_four = C2f(
            in_channels=int(512 * width_multiple),
            out_channels=int(512 * width_multiple * ratio),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.detect_one = Detect(
            in_channels=int(256 * width_multiple),
            out_channels=int(256 * width_multiple),
        )
        self.detect_two = Detect(
            in_channels=int(512 * width_multiple),
            out_channels=int(512 * width_multiple),
        )
        self.detect_three = Detect(
            in_channels=int(512 * width_multiple * ratio),
            out_channels=int(512 * width_multiple * ratio),
        )

    def forward(self, data):
        backbone = self.backbone(data)
        backbone_c2f_two, backbone_c2f_three, backbone_sppf = backbone
        upsample_one = self.upsample_one(backbone_sppf)
        concat_one = torch.concat([upsample_one, backbone_c2f_three])
        c2f_one = self.c2f_one(concat_one)
        upsample_two = self.upsample_two(c2f_one)
        concat_two = torch.concat([upsample_two, backbone_c2f_two])
        c2f_two = self.c2f_two(concat_two)
        conv_one = self.conv_one(c2f_two)
        concat_three = torch.concat([conv_one, c2f_one])
        c2f_three = self.c2f_three(concat_three)
        conv_two = self.conv_two(c2f_three)
        concat_four = torch.concat([conv_two, backbone_sppf])
        c2f_four = self.c2f_four(concat_four)
        detect_one = self.detect_one(c2f_two)
        detect_two = self.detect_two(c2f_three)
        detect_three = self.detect_three(c2f_four)
        return detect_one, detect_two, detect_three
    

MODEL = Head()
print(summary(MODEL, (16, 3, 640, 640)))
