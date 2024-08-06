import keras

depth_multiple, width_multiple, ratio = 1.0, 1.25, 1.0


class Conv(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.block = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding
                ),
                keras.layers.BatchNormalization()
            ]
        )

    def call(self, data):
        return keras.activations.silu(self.block(data))


class SPPF(keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.conv_blocks = [
            Conv(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="valid",
            )
            for _
            in range(2)
        ]
        self.maxpool2d_blocks = [
            keras.layers.MaxPool2D(
                pool_size=1,
                strides=1,
            )
            for _
            in range(3)
        ]

    def call(self, data):
        conv_block_one_output = self.conv_blocks[0](data)
        maxpool2d_block_one_output = self.maxpool2d_blocks[0](conv_block_one_output)
        maxpool2d_block_two_output = self.maxpool2d_blocks[1](maxpool2d_block_one_output)
        maxpool2d_block_three_output = self.maxpool2d_blocks[2](maxpool2d_block_two_output)
        concatenated_output = keras.layers.concatenate(
            [
                conv_block_one_output,
                maxpool2d_block_one_output,
                maxpool2d_block_two_output,
                maxpool2d_block_three_output
            ],
        )
        conv_block_two_output = self.conv_blocks[1](concatenated_output)
        return conv_block_two_output


class BottleNeck(keras.Model):
    def __init__(self, filters, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.conv_blocks_one = Conv(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same"
        )
        self.conv_blocks_two = Conv(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same"
        )

    def call(self, data):
        conv_block_one_output = self.conv_blocks_one(data)
        conv_block_two_output = self.conv_blocks_two(conv_block_one_output)
        if self.shortcut is True:
            return keras.layers.concatenate([data, conv_block_two_output])
        else:
            return conv_block_two_output


class C2f(keras.Model):
    def __init__(self, filters, n, shortcut=True):
        super().__init__()
        self.n = n
        self.shortcut = shortcut
        self.conv_block_one = Conv(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="valid"
        )
        self.conv_block_two = Conv(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="valid"
        )
        self.bottleneck_blocks = [
            BottleNeck(
                filters=filters,
                shortcut=shortcut
            )
            for _
            in range(n)
        ]

    def call(self, data):
        conv_block_one_output = self.conv_block_one(data)
        splitted_output = keras.ops.split(conv_block_one_output, 2, 3)
        bottleneck_block_starter_output = self.bottleneck_blocks[0](splitted_output[1])
        bottleneck_blocks_outputs = bottleneck_block_starter_output
        for index, bottleneck_block in enumerate(self.bottleneck_blocks):
            if index == 0:
                pass
            else:
                bottleneck_blocks_outputs = bottleneck_block(bottleneck_blocks_outputs)
        concatenated_output = keras.layers.concatenate(
            [
                splitted_output[0],
                splitted_output[1],
                bottleneck_block_starter_output,
                bottleneck_blocks_outputs
            ]
        )
        conv_block_two_output = self.conv_block_two(concatenated_output)
        return conv_block_two_output


class Detect(keras.Model):
    def __init__(self, filters, reg_max=8, nc=1):
        super().__init__()
        self.conv = Conv(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.bbox_loss_conv2d = keras.layers.Conv2D(
            filters=4 * reg_max,
            kernel_size=1,
            strides=1,
            padding="valid",
        )
        self.cls_loss_conv2d = keras.layers.Conv2D(
            filters=nc,
            kernel_size=1,
            strides=1,
            padding="valid",
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

    def call(self, data):
        bbox_loss = self.bbox_loss_predictor(data)
        cls_loss = self.cls_loss_predictor(data)
        return [bbox_loss, cls_loss]


class Backbone(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_one = Conv(
            filters=int(64 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.conv_two = Conv(
            filters=int(128 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_one = C2f(
            filters=int(128 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=True,
        )
        self.conv_three = Conv(
            filters=int(256 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_two = C2f(
            filters=int(256 * width_multiple),
            n=int(6 * depth_multiple),
            shortcut=True,
        )
        self.conv_four = Conv(
            filters=int(512 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_three = C2f(
            filters=int(512 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=True,
        )
        self.conv_five = Conv(
            filters=int(512 * width_multiple * ratio),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_four = C2f(
            filters=int(512 * width_multiple * ratio),
            n=int(3 * depth_multiple),
            shortcut=True,
        )
        self.sppf = SPPF(
            filters=int(512 * width_multiple * ratio),
        )

    def call(self, data):
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


class Head(keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.upsample_one = keras.layers.UpSampling2D(size=(2, 2))
        self.c2f_one = C2f(
            filters=int(256 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.upsample_two = keras.layers.UpSampling2D(size=(2, 2))
        self.c2f_two = C2f(
            filters=int(256 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.conv_one = Conv(
            filters=int(256 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_three = C2f(
            filters=int(512 * width_multiple),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.conv_two = Conv(
            filters=int(512 * width_multiple),
            kernel_size=3,
            strides=2,
            padding="same",
        )
        self.c2f_four = C2f(
            filters=int(512 * width_multiple * ratio),
            n=int(3 * depth_multiple),
            shortcut=False,
        )
        self.detect_one = Detect(
            filters=int(256 * width_multiple),
        )
        self.detect_two = Detect(
            filters=int(512 * width_multiple),
        )
        self.detect_three = Detect(
            filters=int(512 * width_multiple * ratio),
        )

    def call(self, data):
        backbone = self.backbone(data)
        backbone_c2f_two, backbone_c2f_three, backbone_sppf = backbone
        upsample_one = self.upsample_one(backbone_sppf)
        concat_one = keras.layers.concatenate([upsample_one, backbone_c2f_three])
        c2f_one = self.c2f_one(concat_one)
        upsample_two = self.upsample_two(c2f_one)
        concat_two = keras.layers.concatenate([upsample_two, backbone_c2f_two])
        c2f_two = self.c2f_two(concat_two)
        conv_one = self.conv_one(c2f_two)
        concat_three = keras.layers.concatenate([conv_one, c2f_one])
        c2f_three = self.c2f_three(concat_three)
        conv_two = self.conv_two(c2f_three)
        concat_four = keras.layers.concatenate([conv_two, backbone_sppf])
        c2f_four = self.c2f_four(concat_four)
        detect_one = self.detect_one(c2f_two)
        detect_two = self.detect_two(c2f_three)
        detect_three = self.detect_three(c2f_four)
        return detect_one, detect_two, detect_three


INPUT = keras.layers.Input((640, 640, 3), batch_size=16)
OUTPUT = Head()(INPUT)
MODEL = keras.Model(inputs=[INPUT], outputs=[OUTPUT])
MODEL.summary()
