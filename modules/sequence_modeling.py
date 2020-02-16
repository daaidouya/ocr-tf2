import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BidirectionalLSTM(keras.Model):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = layers.LSTMCell(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = layers.Dense(hidden_size * 2, output_size)

    def call(self, inputs):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inputs)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
