import tensorflow as tf
from tensorflow.keras import layers, Sequential

# from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import LSTM_SequenceModeling
# from modules.prediction import Attention


class Model(tf.keras.Model):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            # TODO
            # self.Transformation = TPS_SpatialTransformerNetwork(
            #     F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
            #     I_channel_num=opt.inputs_channel)
            print('TPS')
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')

        # Todo self.AdaptiveAvgPool = layers.AdaptiveAvgPool2D((None, 1))  # Transform final (imgH/16-1) -> 1
        # self.AdaptiveAvgPool = layers.AvgPool2D((None, 1))  # Transform final (imgH/16-1) -> 1
        self.Reshape = layers.Reshape((-1, opt.output_channel))

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = LSTM_SequenceModeling(hidden_size=opt.hidden_size)
        else:
            print('No SequenceModeling module specified')

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = layers.Dense(units=opt.num_class)
        elif opt.Prediction == 'Attn':
            # TODO
            # self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
            print('Attn')
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def call(self, inputs, training=None, mask=None):
        """ Transformation stage """
        # if not self.stages['Trans'] == "None":
        #     inputs = self.Transformation(inputs)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(inputs)

        # TODO
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # visual_feature = tf.squeeze(visual_feature)
        visual_feature = self.Reshape(visual_feature)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature)
        else:
            # TODO
            # prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
            #                              batch_max_length=self.opt.batch_max_length)
            print('ATTN')

        return prediction
