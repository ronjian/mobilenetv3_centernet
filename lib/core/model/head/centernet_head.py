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

                deconv_feature = self._unet_magic(fms)

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
            x, y, z,l = tf.split(fm, num_or_size_splits=4, axis=3)

            x = slim.max_pool2d(x,kernel_size=3,stride=1,padding='SAME')
            x = slim.separable_conv2d(x, dim // 4, kernel_size=[3, 3], stride=1, scope='branchx_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            y = slim.conv2d(y, dim // 4, kernel_size=[1, 1], stride=1, scope='branchy_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            z = slim.separable_conv2d(z, dim // 4, kernel_size=[3, 3], stride=1, scope='branchz_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            l= slim.separable_conv2d(l, dim // 4, kernel_size=[5, 5], stride=1, scope='branchse_5x5_pre',
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=None,
                                       biases_initializer=tf.initializers.constant(0.),
                                       )

            final = tf.concat([x, y, z, l], axis=3)  ###96 dims

            return final

    def _complex_upsample(self,fm,dim, scope='upsample'):
        with tf.variable_scope(scope):
            x, y= tf.split(fm, num_or_size_splits=2, axis=3)

            x = self._upsample(x, dim=dim // 2, k_size=3, scope='branch_x_upsample_resize')
            # y = self._upsample_deconv(y,dim=dim//2,scope='branch_y_upsample_deconv')
            y = self._upsample(y, dim=dim // 2, k_size=5, scope='branch_y_upsample_resize')
            final = tf.concat([x, y], axis=3)  ###2*dims

            return final

    def _upsample(self, fm, k_size=5, dim=256, scope='upsample'):

        upsampled_conv = slim.separable_conv2d(fm, dim, [k_size, k_size], padding='SAME', scope=scope)

        upsampled_conv = tf.keras.layers.UpSampling2D(data_format='channels_last')(upsampled_conv)

        return upsampled_conv

    def _upsample_deconv(self, fm,dim, scope='upsample'):
        upsampled_conv = slim.conv2d_transpose(fm, dim, [4, 4], stride=2, padding='SAME', scope=scope)

        return upsampled_conv

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

    def _unet_magic(self, fms, dim=128):

        c2, c3, c4, c5 = fms

        c5_upsample = self._complex_upsample(c5, dim=dim*2, scope='c5_upsample')
        c4 = slim.conv2d(c4, dim*2, [1, 1], padding='SAME', scope='c4_1x1')
        p4 = c4+c5_upsample

        c4_upsample = self._complex_upsample(p4, dim=dim*3//2, scope='c4_upsample')
        c3 = slim.conv2d(c3, dim*3//2, [1, 1], padding='SAME', scope='c3_1x1')
        p3 = c3+c4_upsample

        c3_upsample = self._complex_upsample(p3, dim=dim, scope='c3_upsample')
        c2 = slim.conv2d(c2, dim, [1, 1], padding='SAME', scope='c2_1x1')
        combine_fm = c2+c3_upsample

        return combine_fm

    def _shuffle(self,fm,dim,group=3):

        shape = tf.shape(fm)
        batch_size = shape[0]
        height, width = shape[1], shape[2]

        depth = dim // 3

        ##shufflnet
        if cfg.MODEL.deployee:
            fm = tf.reshape(fm, [height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

            fm = tf.transpose(fm, [0, 1, 3, 2])

        else:
            fm = tf.reshape(fm,
                               [batch_size, height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

            fm = tf.transpose(fm, [0, 1, 2, 4, 3])

        fm = tf.reshape(fm, [batch_size, height, width, group * depth])
        return fm



class CenternetHeadLight():

    def __call__(self, fms, L2_reg, training=True):
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5

                # for i in range(3):
                #     deconv_feature=self._upsample(deconv_feature,scope='upsample_%d'%i)

                deconv_feature = self._unet_magic(fms)

                prefeature = self._pre_head(deconv_feature, 32, 'centernet_pre')

                kps = slim.conv2d(prefeature,
                                  cfg.DATA.num_class,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(-2.19),
                                  scope='centernet_cls_output')



                wh = slim.conv2d(prefeature,
                                 2,
                                 [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0.),
                                 scope='centernet_wh_output')


                reg = slim.conv2d(prefeature,
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
            x, y, z,l = tf.split(fm, num_or_size_splits=4, axis=3)

            x = slim.max_pool2d(x,kernel_size=3,stride=1,padding='SAME')
            x = slim.separable_conv2d(x, dim // 4, kernel_size=[3, 3], stride=1, scope='branchx_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            y = slim.conv2d(y, dim // 4, kernel_size=[1, 1], stride=1, scope='branchy_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            z = slim.separable_conv2d(z, dim // 4, kernel_size=[3, 3], stride=1, scope='branchz_3x3_pre',
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=None,
                                      biases_initializer=tf.initializers.constant(0.),
                                      )

            l= slim.separable_conv2d(l, dim // 4, kernel_size=[5, 5], stride=1, scope='branchse_5x5_pre',
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=None,
                                       biases_initializer=tf.initializers.constant(0.),
                                       )

            final = tf.concat([x, y, z, l], axis=3)  ###96 dims

            return final

    def _complex_upsample(self,fm,dim, scope='upsample'):
        with tf.variable_scope(scope):
            x, y= tf.split(fm, num_or_size_splits=2, axis=3)

            x = self._upsample(x, dim=dim // 2, k_size=3, scope='branch_x_upsample_resize')
            y=self._upsample_deconv(y,dim=dim//2,scope='branch_y_upsample_deconv')

            final = tf.concat([x, y], axis=3)  ###2*dims

            return final

    def _upsample(self, fm, k_size=5, dim=256, scope='upsample'):

        upsampled_conv = slim.separable_conv2d(fm, dim, [k_size, k_size], padding='SAME', scope=scope)

        upsampled_conv = tf.keras.layers.UpSampling2D(data_format='channels_last')(upsampled_conv)

        return upsampled_conv

    def _upsample_deconv(self, fm,dim, scope='upsample'):
        upsampled_conv = slim.conv2d_transpose(fm, dim, [4, 4], stride=2, padding='SAME', scope=scope)

        return upsampled_conv

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

    def _unet_magic(self, fms, dim=64):

        c2, c3, c4, c5 = fms

        c5_upsample = self._complex_upsample(c5, dim=dim, scope='c5_upsample')
        c4 = slim.conv2d(c4, dim, [1, 1], padding='SAME', scope='c4_1x1')
        p4 = c4+c5_upsample

        c4_upsample = self._complex_upsample(p4, dim=48, scope='c4_upsample')
        c3 = slim.conv2d(c3, 48, [1, 1], padding='SAME', scope='c3_1x1')
        p3 = c3+c4_upsample

        c3_upsample = self._complex_upsample(p3, dim=32, scope='c3_upsample')
        c2 = slim.conv2d(c2, 32, [1, 1], padding='SAME', scope='c2_1x1')
        combine_fm = c2+c3_upsample

        return combine_fm

    def _shuffle(self,fm,dim,group=3):

        shape = tf.shape(fm)
        batch_size = shape[0]
        height, width = shape[1], shape[2]

        depth = dim // 3

        ##shufflnet
        if cfg.MODEL.deployee:
            fm = tf.reshape(fm, [height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

            fm = tf.transpose(fm, [0, 1, 3, 2])

        else:
            fm = tf.reshape(fm,
                               [batch_size, height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

            fm = tf.transpose(fm, [0, 1, 2, 4, 3])

        fm = tf.reshape(fm, [batch_size, height, width, group * depth])
        return fm