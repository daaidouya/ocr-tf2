import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np


class TPS_SpatialTransformerNetwork(tf.keras.Model):
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        # self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def call(self, batch_I):
        # TODO
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = tf.layers.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(tf.keras.Model):
    def __init__(self, num_fiducial, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = Sequential(
            layers.Conv2D(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), layers.BatchNormalization(64), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            layers.Conv2D(64, 128, 3, 1, 1, bias=False), layers.BatchNormalization(128), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            layers.Conv2D(128, 256, 3, 1, 1, bias=False), layers.BatchNormalization(256), layers.ReLU(True),
            layers.MaxPool2D(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            layers.Conv2D(256, 512, 3, 1, 1, bias=False), layers.BatchNormalization(512), layers.ReLU(True),
            # TODO
            layers.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = Sequential(layers.Dense(512, 256), layers.ReLU(True))
        self.localization_fc2 = layers.Dense(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = tf.convert_to_tensor(initial_bias)#.float().view(-1)

    def call(self, batch_I):
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime

