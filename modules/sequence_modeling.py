import tensorflow as tf
from tensorflow.keras import layers, Sequential


class LSTM_SequenceModeling(tf.keras.Model):

    def __init__(self, hidden_size):
        super(LSTM_SequenceModeling, self).__init__()
        # x = keras.layers.Reshape((-1, 512))(x)
        # x = keras.layers.Bidirectional(
        #     keras.layers.LSTM(units=256, return_sequences=True))(x)
        # x = keras.layers.Bidirectional(
        #     keras.layers.LSTM(units=256, return_sequences=True))(x)

        self.rnn1 = layers.Bidirectional(layers.LSTM(units=hidden_size, return_sequences=True))
        # self.linear = nn.Linear(hidden_size * 2, output_size)
        self.linear1 = layers.Dense(units=hidden_size)

        self.rnn2 = layers.Bidirectional(layers.LSTM(units=hidden_size, return_sequences=True))
        # self.linear = nn.Linear(hidden_size * 2, output_size)
        self.linear2 = layers.Dense(units=hidden_size)

    def call(self, inputs, training=None, mask=None):
        out = self.rnn1(inputs)
        out = self.linear1(out)
        out = self.rnn2(out)
        out = self.linear2(out)
        return out
