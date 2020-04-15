# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from tensorflow.python.ops.init_ops import Initializer

import numpy as np
import math


class CenternetHead():

    def __call__(self, fms, L2_reg, training=True):
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5

                # for i in range(3):
                #     deconv_feature=self._upsample(deconv_feature,scope='upsample_%d'%i)

                deconv_feature = self._unet_inception(fms)

                kps = self._pre_head(deconv_feature, 128, 'centernet_cls_pre')
                kps = slim.conv2d(kps,
                                cfg.DATA.num_class,
                                [1, 1],
                                stride=1,
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                biases_initializer=tf.initializers.constant(-2.19),
                                scope='centernet_cls_output')
                
                wh = self._pre_head(deconv_feature, 64, 'centernet_wh_pre')

                wh = slim.conv2d(wh,
                               2,
                               [1, 1],
                               stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               weights_initializer=tf.initializers.random_normal(stddev=0.001),
                               biases_initializer=tf.initializers.constant(0.),
                               scope='centernet_wh_output')

                reg = self._pre_head(deconv_feature, 64, 'centernet_reg_pre')
                reg = slim.conv2d(reg,
                                2,
                                [1, 1],
                                stride=1,
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                biases_initializer=tf.initializers.constant(0.),
                                scope='centernet_reg_output')
        return kps, wh, reg

    def _pre_head(self, fm, dim, scope):
        with tf.variable_scope(scope):
            x, y, z, se = tf.split(fm, num_or_size_splits=4, axis=3)

            x = slim.max_pool2d(x, kernel_size=3, stride=1)

            y = slim.conv2d(y, dim // 4, kernel_size=[1, 1], stride=1, scope='branchy_1x1_pre')

            z = slim.separable_conv2d(z, dim // 4, kernel_size=[3, 3], stride=1, scope='branchz_3x3_pre')

            se = slim.separable_conv2d(se, dim // 4, kernel_size=[3, 3], stride=1, scope='branchse_3x3_pre')
            se = slim.separable_conv2d(se, dim // 4, kernel_size=[3, 3], stride=1, scope='branchse_3x3_after')

            final = tf.concat([x, y, z, se], axis=3)  ###96 dims

            return final

    def _upsample(self, fm, k_size=5, dim=256, scope='upsample'):
        upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(fm)

        if k_size == 1:
            upsampled_conv = slim.conv2d(upsampled, dim, [k_size, k_size], padding='SAME', scope=scope)
        else:
            upsampled_conv = slim.separable_conv2d(upsampled, dim, [k_size, k_size], padding='SAME', scope=scope)
        return upsampled_conv

    def _upsample_deconv(self, fm, scope='upsample'):
        upsampled_conv = slim.conv2d_transpose(fm, 256, [4, 4], stride=2, padding='SAME', scope=scope)
        return upsampled_conv

    def _inception_upsample(self, fm, dim, scope):
        with tf.variable_scope(scope):
            x, y, z, se = tf.split(fm, num_or_size_splits=4, axis=3)

            x = self._upsample(x, dim=dim // 4, k_size=1, scope='branch_x_upsample')

            y = slim.conv2d(y, dim // 4, kernel_size=[1, 1], stride=1, scope='branchy_1x1_pre')
            y = self._upsample(y, dim=dim // 4, k_size=3, scope='branch_y_upsample')

            z = slim.separable_conv2d(z, dim // 4, kernel_size=[3, 3], stride=1, scope='branchz_3x3_pre')
            z = self._upsample(z, dim=dim // 4, k_size=5, scope='branch_z_upsample')

            se = tf.reduce_mean(se, axis=[1, 2], keep_dims=True)
            se = slim.conv2d(se, dim // 4, kernel_size=[1, 1], scope='se_model')
            z = z * se

            final = tf.concat([x, y, z], axis=3)  ###96 dims

            return final

    def _unet_upsample(self, fms):
        c2, c3, c4, c5 = fms

        c5_upsample = self._upsample(c5, k_size=5, dim=128, scope='c5_upsample')

        c4 = slim.separable_conv2d(c4, 128, [3, 3], padding='SAME', scope='c4_1x1')
        p4 = tf.concat([c4, c5_upsample], axis=3)
        c4_upsample = self._upsample(p4, k_size=5, dim=128, scope='c4_upsample')

        c3 = slim.separable_conv2d(c3, 128, [3, 3], padding='SAME', scope='c3_1x1')
        p3 = tf.concat([c3, c4_upsample], axis=3)
        c3_upsample = self._upsample(p3, k_size=5, dim=128, scope='c3_upsample')

        c2 = slim.separable_conv2d(c2, 128, [3, 3], padding='SAME', scope='c2_1x1')
        combine_fm = tf.concat([c2, c3_upsample], axis=3)

        combine_fm = slim.separable_conv2d(combine_fm, 256, [3, 3], padding='SAME', scope='combine_fm')
        return combine_fm

    def _unet_inception(self, fms, dim=32):
        c2, c3, c4, c5 = fms

        c5_upsample = self._inception_upsample(c5, dim=dim * 4, scope='c5_upsample')

        c4 = slim.conv2d(c4, dim, [1, 1], padding='SAME', scope='c4_1x1')
        p4 = tf.concat([c4, c5_upsample], axis=3)
        c4_upsample = self._inception_upsample(p4, dim=dim * 4, scope='c4_upsample')

        c3 = slim.conv2d(c3, dim, [1, 1], padding='SAME', scope='c3_1x1')
        p3 = tf.concat([c3, c4_upsample], axis=3)
        c3_upsample = self._inception_upsample(p3, dim=dim * 4, scope='c3_upsample')

        c2 = slim.conv2d(c2, dim, [1, 1], padding='SAME', scope='c2_1x1')
        combine_fm = tf.concat([c2, c3_upsample], axis=3)

        # combine_fm = slim.separable_conv2d(combine_fm, 128, [3, 3], padding='SAME', scope='combine_fm')
        return combine_fm


class CenternetHeadLight():

    def __call__(self, fms, L2_reg, training=True):
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                c3, c4, c5 = fms
                deconv_feature = c5
                for i in range(3):
                    deconv_feature = self._upsample(deconv_feature, scope='upsample_%d' % i)

                kps = slim.conv2d(deconv_feature,
                                  cfg.DATA.num_class,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(-2.19),
                                  scope='centernet_cls_output')

                wh = slim.conv2d(deconv_feature,
                                 2,
                                 [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0.),
                                 scope='centernet_wh_output')

                reg = slim.conv2d(deconv_feature,
                                  2,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(0.),
                                  scope='centernet_reg_output')
        return kps, wh, reg

    def _upsample(self, fm, scope='upsample'):
        upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last', interpolation='bilinear')(fm)
        upsampled_conv = slim.separable_conv2d(upsampled, 256, [1, 1], padding='SAME', scope=scope)
        return upsampled_conv

    def _upsample_deconv(self, fm, scope='upsample'):
        upsampled_conv = slim.conv2d_transpose(fm, 256, [4, 4], stride=2, padding='SAME', scope=scope)
        return upsampled_conv