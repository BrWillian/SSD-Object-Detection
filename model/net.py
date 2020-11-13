from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D, Convolution2D
from tensorflow.keras.layers import concatenate, Flatten, Reshape, Activation
from model.activations import Mish
from model.layers import Normalize, PriorBox
from tensorflow.keras import backend as K


class SSD512:
    def __init__(self, **kwargs):
        self.net = {}
        super(SSD512, self).__init__(**kwargs)

    def build(self, input_shape=(300, 300, 3), num_classes=1):
        self.net['inputs'] = Input(shape=input_shape, name='inputs')

        self.net['conv1_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_1_zp')(self.net['inputs'])
        self.net['conv1_1'] = Convolution2D(64, (3, 3), activation=Mish(), strides=(1, 1), name='conv1_1')(self.net['conv1_1_zp'])
        self.net['conv1_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_2_zp')(self.net['conv1_1'])
        self.net['conv1_2'] = Convolution2D(64, (3, 3), activation=Mish(), strides=(1, 1), name='conv1_2')(self.net['conv1_2_zp'])
        self.net['pool1'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(self.net['conv1_2'])

        self.net['conv2_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_1_zp')(self.net['pool1'])
        self.net['conv2_1'] = Convolution2D(128, (3, 3), activation=Mish(), strides=(1, 1), name='conv2_1')(self.net['conv2_1_zp'])
        self.net['conv2_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_2_zp')(self.net['conv2_1'])
        self.net['conv2_2'] = Convolution2D(128, (3, 3), activation=Mish(), strides=(1, 1), name='conv2_2')(self.net['conv2_2_zp'])
        self.net['pool2'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(self.net['conv2_2'])

        self.net['conv3_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_1_zp')(self.net['pool2'])
        self.net['conv3_1'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(1, 1), name='conv3_1')(self.net['conv3_1_zp'])
        self.net['conv3_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_2_zp')(self.net['conv3_1'])
        self.net['conv3_2'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(1, 1), name='conv3_2')(self.net['conv3_2_zp'])
        self.net['conv3_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_3_zp')(self.net['conv3_2'])
        self.net['conv3_3'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(1, 1), name='conv3_3')(self.net['conv3_3_zp'])
        self.net['pool3'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(self.net['conv3_3'])

        self.net['conv4_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_1_zp')(self.net['pool3'])
        self.net['conv4_1'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv4_1')(self.net['conv4_1_zp'])
        self.net['conv4_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_2_zp')(self.net['conv4_1'])
        self.net['conv4_2'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv4_2')(self.net['conv4_2_zp'])
        self.net['conv4_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_zp')(self.net['conv4_2'])
        self.net['conv4_3'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv4_3')(self.net['conv4_3_zp'])
        self.net['pool4'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(self.net['conv4_3'])

        self.net['conv5_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_1_zp')(self.net['pool4'])
        self.net['conv5_1'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv5_1')(self.net['conv5_1_zp'])
        self.net['conv5_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_2_zp')(self.net['conv5_1'])
        self.net['conv5_2'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv5_2')(self.net['conv5_2_zp'])
        self.net['conv5_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_3_zp')(self.net['conv5_2'])
        self.net['conv5_3'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(1, 1), name='conv5_3')(self.net['conv5_3_zp'])
        self.net['pool5_zp'] = ZeroPadding2D(padding=(1, 1), name='pool5_zp')(self.net['conv5_3'])
        self.net['pool5'] = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), name='pool5')(self.net['pool5_zp'])

        self.net['fc6_zp'] = ZeroPadding2D(padding=(6, 6), name='fc6_zp')(self.net['pool5'])
        self.net['fc6'] = Convolution2D(1024, (3, 3), activation=Mish(), strides=(1, 1), dilation_rate=(6, 6), name='fc6')(self.net['fc6_zp'])

        self.net['fc7'] = Convolution2D(1024, (1, 1), activation=Mish(), strides=(1, 1), name='fc7')(self.net['fc6'])

        self.net['conv6_1'] = Convolution2D(256, (1, 1), activation=Mish(), strides=(1, 1), name='conv6_1')(self.net['fc7'])
        self.net['conv6_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_zp')(self.net['conv6_1'])
        self.net['conv6_2'] = Convolution2D(512, (3, 3), activation=Mish(), strides=(2, 2), name='conv6_2')(self.net['conv6_2_zp'])

        self.net['conv7_1'] = Convolution2D(128, (1, 1), activation=Mish(), strides=(1, 1), name='conv7_1')(self.net['conv6_2'])
        self.net['conv7_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_zp')(self.net['conv7_1'])
        self.net['conv7_2'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(2, 2), name='conv7_2')(self.net['conv7_2_zp'])

        self.net['conv8_1'] = Convolution2D(128, (1, 1), activation=Mish(), strides=(1, 1), name='conv8_1')(self.net['conv7_2'])
        self.net['conv8_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_zp')(self.net['conv8_1'])
        self.net['conv8_2'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(2, 2), name='conv8_2')(self.net['conv8_2_zp'])

        self.net['conv9_1'] = Convolution2D(128, (1, 1), activation=Mish(), strides=(1, 1), name='conv9_1')(self.net['conv8_2'])
        self.net['conv9_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_zp')(self.net['conv9_1'])
        self.net['conv9_2'] = Convolution2D(256, (3, 3), activation=Mish(), strides=(2, 2), name='conv9_2')(self.net['conv9_2_zp'])

        self.net['conv10_1'] = Convolution2D(128, (1, 1), activation=Mish(), strides=(1, 1), name='conv10_1')(self.net['conv9_2'])
        self.net['conv10_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_zp')(self.net['conv10_1'])
        self.net['conv10_2'] = Convolution2D(256, (4, 4), activation=Mish(), strides=(1, 1), name='conv10_2')(self.net['conv10_2_zp'])

        self.net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(self.net['conv4_3'])

        num_priors = 4
        self.net['conv4_3_norm_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_loc_zp')(self.net['conv4_3_norm'])
        self.net['conv4_3_norm_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv4_3_norm_mbox_loc')(self.net['conv4_3_norm_mbox_loc_zp'])
        self.net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(self.net['conv4_3_norm_mbox_loc'])
        self.net['conv4_3_norm_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_conf_zp')(self.net['conv4_3_norm'])
        self.net['conv4_3_norm_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1),name='conv4_3_norm_mbox_conf')(self.net['conv4_3_norm_mbox_conf_zp'])
        self.net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(self.net['conv4_3_norm_mbox_conf'])
        self.net['conv4_3_norm_mbox_priorbox'] = PriorBox((512, 512), min_size=35.84, max_size=76.80,aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],flip=True, clip=False, name='conv4_3_norm_mbox_priorbox')(self.net['conv4_3_norm'])

        num_priors = 6
        self.net['fc7_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_loc_zp')(self.net['fc7'])
        self.net['fc7_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='fc7_mbox_loc')(self.net['fc7_mbox_loc_zp'])
        self.net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(self.net['fc7_mbox_loc'])
        self.net['fc7_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_conf_zp')(self.net['fc7'])
        self.net['fc7_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='fc7_mbox_conf')(self.net['fc7_mbox_conf_zp'])
        self.net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(self.net['fc7_mbox_conf'])
        self.net['fc7_mbox_priorbox'] = PriorBox((512, 512), min_size=76.80, max_size=153.60,aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],flip=True, clip=False, name='fc7_mbox_priorbox')(self.net['fc7'])

        self.net['conv6_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_loc_zp')(self.net['conv6_2'])
        self.net['conv6_2_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv6_2_mbox_loc')(self.net['conv6_2_mbox_loc_zp'])
        self.net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(self.net['conv6_2_mbox_loc'])
        self.net['conv6_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_conf_zp')(self.net['conv6_2'])
        self.net['conv6_2_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv6_2_mbox_conf')(self.net['conv6_2_mbox_conf_zp'])
        self.net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(self.net['conv6_2_mbox_conf'])
        self.net['conv6_2_mbox_priorbox'] = PriorBox((512, 512), min_size=153.60, max_size=230.40, aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],flip=True, clip=False, name='conv6_2_mbox_priorbox')(self.net['conv6_2'])

        self.net['conv7_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_loc_zp')(self.net['conv7_2'])
        self.net['conv7_2_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv7_2_mbox_loc')(self.net['conv7_2_mbox_loc_zp'])
        self.net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(self.net['conv7_2_mbox_loc'])
        self.net['conv7_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_conf_zp')(self.net['conv7_2'])
        self.net['conv7_2_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv7_2_mbox_conf')(self.net['conv7_2_mbox_conf_zp'])
        self.net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(self.net['conv7_2_mbox_conf'])
        self.net['conv7_2_mbox_priorbox'] = PriorBox((512, 512), min_size=230.40, max_size=307.20,aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],flip=True, clip=False, name='conv7_2_mbox_priorbox')(self.net['conv7_2'])

        self.net['conv8_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_loc_zp')(self.net['conv8_2'])
        self.net['conv8_2_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv8_2_mbox_loc')(self.net['conv8_2_mbox_loc_zp'])
        self.net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(self.net['conv8_2_mbox_loc'])
        self.net['conv8_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_conf_zp')(self.net['conv8_2'])
        self.net['conv8_2_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv8_2_mbox_conf')(self.net['conv8_2_mbox_conf_zp'])
        self.net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(self.net['conv8_2_mbox_conf'])
        self.net['conv8_2_mbox_priorbox'] = PriorBox((512, 512), min_size=307.20, max_size=384.00, aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],flip=True, clip=False, name='conv8_2_mbox_priorbox')(self.net['conv8_2'])

        num_priors = 4
        self.net['conv9_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_loc_zp')(self.net['conv9_2'])
        self.net['conv9_2_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv9_2_mbox_loc')(self.net['conv9_2_mbox_loc_zp'])
        self.net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(self.net['conv9_2_mbox_loc'])
        self.net['conv9_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_conf_zp')(self.net['conv9_2'])
        self.net['conv9_2_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv9_2_mbox_conf')(self.net['conv9_2_mbox_conf_zp'])
        self.net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(self.net['conv9_2_mbox_conf'])
        self.net['conv9_2_mbox_priorbox'] = PriorBox((512, 512), min_size=384.00, max_size=460.80, aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20], flip=True, clip=False, name='conv9_2_mbox_priorbox')(self.net['conv9_2'])

        self.net['conv10_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_mbox_loc_zp')(self.net['conv10_2'])
        self.net['conv10_2_mbox_loc'] = Convolution2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv10_2_mbox_loc')(self.net['conv10_2_mbox_loc_zp'])
        self.net['conv10_2_mbox_loc_flat'] = Flatten(name='conv10_2_mbox_loc_flat')(self.net['conv10_2_mbox_loc'])
        self.net['conv10_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_mbox_conf_zp')(self.net['conv10_2'])
        self.net['conv10_2_mbox_conf'] = Convolution2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv10_2_mbox_conf')(self.net['conv10_2_mbox_conf_zp'])
        self.net['conv10_2_mbox_conf_flat'] = Flatten(name='conv10_2_mbox_conf_flat')(self.net['conv10_2_mbox_conf'])
        self.net['conv10_2_mbox_priorbox'] = PriorBox((512, 512), min_size=460.80, max_size=537.60, aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20], flip=True, clip=False, name='conv10_2_mbox_priorbox')(self.net['conv10_2'])

        self.net['mbox_loc'] = concatenate(inputs=[self.net['conv4_3_norm_mbox_loc_flat'],
                                              self.net['fc7_mbox_loc_flat'],
                                              self.net['conv6_2_mbox_loc_flat'],
                                              self.net['conv7_2_mbox_loc_flat'],
                                              self.net['conv8_2_mbox_loc_flat'],
                                              self.net['conv9_2_mbox_loc_flat'],
                                              self.net['conv10_2_mbox_loc_flat']], axis=1, name='mbox_loc')
        self.net['mbox_conf'] = concatenate(inputs=[self.net['conv4_3_norm_mbox_conf_flat'],
                                               self.net['fc7_mbox_conf_flat'],
                                               self.net['conv6_2_mbox_conf_flat'],
                                               self.net['conv7_2_mbox_conf_flat'],
                                               self.net['conv8_2_mbox_conf_flat'],
                                               self.net['conv9_2_mbox_conf_flat'],
                                               self.net['conv10_2_mbox_conf_flat']], axis=1, name='mbox_conf')
        self.net['mbox_priorbox'] = concatenate(inputs=[self.net['conv4_3_norm_mbox_priorbox'],
                                                   self.net['fc7_mbox_priorbox'],
                                                   self.net['conv6_2_mbox_priorbox'],
                                                   self.net['conv7_2_mbox_priorbox'],
                                                   self.net['conv8_2_mbox_priorbox'],
                                                   self.net['conv9_2_mbox_priorbox'],
                                                   self.net['conv10_2_mbox_priorbox']], axis=1, name='mbox_priorbox')


        num_boxes = K.int_shape(self.net['mbox_loc'])[-1] // 4

        self.net['mbox_loc'] = Reshape((num_boxes, 4), name='mbox_loc_final')(self.net['mbox_loc'])
        self.net['mbox_conf'] = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(self.net['mbox_conf'])
        self.net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(self.net['mbox_conf'])
        self.net['predictions'] = concatenate([self.net['mbox_loc'], self.net['mbox_conf'], self.net['mbox_priorbox']], axis=2, name='predictions')

        model = Model(self.net['inputs'], self.net['predictions'])

        return model


model = SSD512()
model.build((512, 512, 3), 2)

