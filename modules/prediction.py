import tensorflow as tf
from tensorflow.keras import layers, Sequential
# TODO
device = tf.device()


class Attention(tf.keras.Model):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        # self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.attention_cell = layers.Attention(use_scale=True)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = layers.Dense(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        # TODO
        one_hot = tf.nn.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def call(self, batch_H, text, is_train=True, batch_max_length=25):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(tf.keras.Model):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = layers.Dense(units=hidden_size, use_bias=False)
        self.h2h = layers.Dense(units=hidden_size, use_bias=False)
        self.score = layers.Dense(units=1, use_bias=False)

        # self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.rnn = layers.LSTMCell(units=hidden_size)()
        self.hidden_size = hidden_size

    def call(self, prev_hidden, batch_H, char_onehots):

        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), axis=1)
        e = self.score(tf.nn.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1
        alpha = tf.nn.softmax(e, axis=1)

        context = tf.squeeze(tf.matmul(tf.transpose(alpha, [0, 2, 1]), batch_H), axis=1)
        concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        # TODO
        cur_hidden = self.rnn(inputs=concat_context, states=prev_hidden)
        return cur_hidden, alpha

