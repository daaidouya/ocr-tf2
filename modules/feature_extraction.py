import tensorflow as tf
from tensorflow.keras import layers, Sequential


class VGG_FeatureExtractor(tf.keras.Model):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = Sequential([
            layers.Conv2D(self.output_channel[0], kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(self.output_channel[1], kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(self.output_channel[2], kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(self.output_channel[2], kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
            layers.Conv2D(self.output_channel[3], kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(), layers.ReLU(),
            layers.Conv2D(self.output_channel[3], kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(), layers.ReLU(),
            layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1)),
            layers.Conv2D(self.output_channel[3], kernel_size=2, strides=1, padding='valid', activation='relu'),
        ])

    def call(self, inputs, training=None, mask=None):
        # self.ConvNet.summary()
        return self.ConvNet(inputs)


class RCNN_FeatureExtractor(tf.keras.Model):
    """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

    def __init__(self, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = Sequential([
            layers.Conv2D(self.output_channel[0], kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=2),

            GRCL(self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),

            layers.MaxPool2D(pool_size=(2, 2), strides=2),

            GRCL(self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),

            # layers.MaxPool2D(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            layers.MaxPool2D(pool_size=(2, 2), strides=2),

            GRCL(self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),

            # layers.MaxPool2D(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            layers.MaxPool2D(pool_size=(2, 2), strides=2),

            layers.Conv2D(self.output_channel[3], kernel_size=2, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.ConvNet(inputs)


class ResNet_FeatureExtractor(tf.keras.Model):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(output_channel, [1, 2, 5, 3])

    def call(self, inputs, training=None, mask=None):
        return self.ConvNet(inputs)


# For Gated RCNN
class GRCL(layers.Layer):

    def __init__(self, output_channel, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = layers.Conv2D(output_channel, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.wgr_x = layers.Conv2D(output_channel, kernel_size=1, strides=1, padding='valid', use_bias=False)

        self.wf_u = layers.Conv2D(output_channel, kernel_size=kernel_size, strides=1, padding=pad, use_bias=False)
        self.wr_x = layers.Conv2D(output_channel, kernel_size=kernel_size, strides=1, padding=pad, use_bias=False)
        self.BN_x_init = layers.BatchNormalization()

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit() for _ in range(num_iteration)]
        self.GRCL = Sequential(self.GRCL)

    def call(self, inputs, **kwargs):
        """ The inputs of GRCL is consistant over time t, which is denoted by u(0)
        thus wgf_u / wf_u is also consistant over time t.
        """
        wgf_u = self.wgf_u(inputs)
        wf_u = self.wf_u(inputs)
        x = tf.nn.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

        return x


class GRCL_unit(layers.Layer):

    def __init__(self):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = layers.BatchNormalization()
        self.BN_grx = layers.BatchNormalization()
        self.BN_fu = layers.BatchNormalization()
        self.BN_rx = layers.BatchNormalization()
        self.BN_Gx = layers.BatchNormalization()

    def call(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = tf.nn.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = tf.nn.relu(x_first_term + x_second_term)

        return x


class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=None):

        # [b, h, w, c]，通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(tf.keras.Model):

    def __init__(self, output_channel, layer_dims):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = layers.Conv2D(int(output_channel / 16), kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn0_1 = layers.BatchNormalization()
        self.conv0_2 = layers.Conv2D(self.inplanes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn0_2 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.maxpool1 = layers.MaxPool2D(pool_size=2, padding='same')
        self.layer1 = self._make_layer(self.output_channel_block[0], layer_dims[0])
        self.conv1 = layers.Conv2D(self.output_channel_block[0], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.maxpool2 = layers.MaxPool2D(pool_size=2, padding='same')
        self.layer2 = self._make_layer(self.output_channel_block[1], layer_dims[1], stride=1)
        self.conv2 = layers.Conv2D(self.output_channel_block[1], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.maxpool3 = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')
        self.layer3 = self._make_layer(self.output_channel_block[2], layer_dims[2], stride=1)
        self.conv3 = layers.Conv2D(self.output_channel_block[2], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.layer4 = self._make_layer(self.output_channel_block[3], layer_dims[3], stride=1)
        # TODO: padding
        self.conv4_1 = layers.Conv2D(self.output_channel_block[3], kernel_size=2, strides=(2, 1), padding='same', use_bias=False)
        self.bn4_1 = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(self.output_channel_block[3], kernel_size=2, strides=1, padding='valid', use_bias=False)
        self.bn4_2 = layers.BatchNormalization()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        # TODO: if stride != 1:
        if True:
            downsample = Sequential([
                layers.Conv2D(planes * 1, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization(),
            ])

        layers_list = [BasicBlock(planes, stride, downsample)]
        self.inplanes = planes * 1
        for _ in range(1, blocks):
            layers_list.append(BasicBlock(planes))

        return Sequential(layers_list)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x

