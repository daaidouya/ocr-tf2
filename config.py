"""
OCR config.py
feature_extraction:
prediction:
sequence_modeling
transformation
"""


class OCR_Config(object):
    def __init__(self):
        self.train_annotation_paths = './example/annotations.txt'
        self.val_annotation_paths = './example/val_annotations.txt'
        self.table_path = './example/characters.txt'  # characters list.
        self.experiment_name = ''
        self.blank_index = 1
        self.batch_size = 8
        self.epochs = 3
        self.checkpoint = './saved_models/None-VGG-None-CTC'
        self.save_freq = 1  # Save and validate interval.
        self.max_to_keep = 5  # Max num of checkpoint to keep.
        """ Optimizer """
        self.adam = False  # Whether to use adam (default is Adadelta)
        self.learning_rate = 1
        self.beta1 = 0.9  # beta1 for adam. default=0.9
        self.rho = 0.95  # decay rate rho for Adadelta. default=0.95
        self.epsilon = 1e-8  # eps for Adadelta. default=1e-8
        """ Data processing """
        self.image_height = 32
        self.image_width = 100
        self.rgb = False
        self.keep_ratio_with_pad = True
        """ Model Architecture """
        self.Transformation = 'None'  # Transformation stage. None|TPS
        self.FeatureExtraction = 'ResNet'  # FeatureExtraction stage. VGG|RCNN|ResNet
        self.SequenceModeling = 'BiLSTM'  # SequenceModeling stage. None|BiLSTM
        self.Prediction = 'CTC'  # Prediction stage. CTC|Attn
        self.output_channel = 512
        self.hidden_size = 256


args = OCR_Config()
