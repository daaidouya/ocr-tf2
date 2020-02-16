import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class ResNet_FeatureExtractor(layers.Model):
    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock)

    def call(self, inputs):
        return self.ConvNet(inputs)


class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        # [b, h, w, c]，通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, num_classes=10): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        # 根网络，预处理
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # 通过Pooling层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # 通过根网络
        x = self.stem(inputs)
        # 一次通过4个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通过池化层
        x = self.avgpool(x)
        # 通过全连接层
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):#其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])


def resnet34():
     # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3])


class VGG_FeatureExtractor(keras.Model):
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = Sequential(
            layers.Conv2D(input_channel, self.output_channel[0], 3, 1, 1), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # 64x16x50
            layers.Conv2D(self.output_channel[0], self.output_channel[1], 3, 1, 1), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # 128x8x25
            layers.Conv2D(self.output_channel[1], self.output_channel[2], 3, 1, 1), layers.ReLU(True),  # 256x8x25
            layers.Conv2D(self.output_channel[2], self.output_channel[2], 3, 1, 1), layers.ReLU(True),
            layers.MaxPool2D((2, 1), (2, 1)),  # 256x4x25
            layers.Conv2D(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            layers.BatchNormalization(self.output_channel[3]), layers.ReLU(True),  # 512x4x25
            layers.Conv2D(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            layers.BatchNormalization(self.output_channel[3]), layers.ReLU(True),
            layers.MaxPool2D((2, 1), (2, 1)),  # 512x2x25
            layers.Conv2D(self.output_channel[3], self.output_channel[3], 2, 1, 0), layers.ReLU(True))  # 512x1x24

    def call(self, inputs):
        return self.ConvNet(inputs)


# For Gated RCNN
class GRCL(layers.Layer):

    def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = layers.Conv2D(input_channel, output_channel, 1, 1, 0, bias=False)
        self.wgr_x = layers.Conv2D(output_channel, output_channel, 1, 1, 0, bias=False)
        self.wf_u = layers.Conv2D(input_channel, output_channel, kernel_size, 1, pad, bias=False)
        self.wr_x = layers.Conv2D(output_channel, output_channel, kernel_size, 1, pad, bias=False)

        self.BN_x_init = layers.BatchNormalization(output_channel)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
        self.GRCL = Sequential(*self.GRCL)

    def call(self, inputs, training=None):

        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = layers.ReLU(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(keras.Model):
    def __init__(self, output_channel):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = layers.BatchNormalization(output_channel)
        self.BN_grx = layers.BatchNormalization(output_channel)
        self.BN_fu = layers.BatchNormalization(output_channel)
        self.BN_rx = layers.BatchNormalization(output_channel)
        self.BN_Gx = layers.BatchNormalization(output_channel)

    def call(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = layers.Sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = layers.ReLU(x_first_term + x_second_term)

        return x


class RCNN_FeatureExtractor(keras.Model):
    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = Sequential(
            layers.Conv2D(input_channel, self.output_channel[0], 3, 1, 1), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            layers.MaxPool2D(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            layers.MaxPool2D(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            layers.MaxPool2D(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            layers.Conv2D(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            layers.BatchNormalization(self.output_channel[3]), layers.ReLU(True))  # 512 x 1 x 26

    def call(self, inputs):
        return self.ConvNet(inputs)
