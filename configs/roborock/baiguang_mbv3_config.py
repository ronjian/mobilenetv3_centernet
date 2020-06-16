#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 4                      ### process_num for data provider
config.TRAIN.prefetch_size = 20                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 16                   ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 1
config.TRAIN.epoch = 300                      ###just keep training , evaluation shoule be take care by yourself,
                                               ### generally 10,0000 iters is enough

config.TRAIN.train_set_size=13112            ###widerface train size
config.TRAIN.val_set_size=1556             ###widerface val size

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.00025,0.000025,0.0000025,0.00000025]        ##warm up is used
config.TRAIN.lr_decay_every_step = [2 * config.TRAIN.train_set_size
                                    ,20 * config.TRAIN.train_set_size
                                    ,50 * config.TRAIN.train_set_size
                                    ,100 * config.TRAIN.train_set_size
                                    ,200 * config.TRAIN.train_set_size]

config.TRAIN.opt='adam'
config.TRAIN.weight_decay_factor = 5.e-5                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag
config.TRAIN.mix_precision=False

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze
config.TRAIN.gradient_clip=False

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='data/baiguang-train.txt'
config.DATA.val_txt_path='data/baiguang-val.txt'
config.DATA.num_category=29                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category

config.DATA.PIXEL_MEAN = [127.]                 ###rgb
config.DATA.PIXEL_STD = [127.]

# config.DATA.hin = 512  # input size
# config.DATA.win = 512
config.DATA.hin = 288  # input size
config.DATA.win = 384
config.DATA.channel = 3
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=0                          ###cover the small faces

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)
config.DATA.use_int8_data=True
config.DATA.use_int8_enlarge=255.           ### use uint8 for heatmap generate for less memory acc, to speed up
config.DATA.max_objs=128

##mobilenetv3 as basemodel
config.MODEL = edict()
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.model_path = './weights/baiguang-mbv3/baiguang-mbv3'  # save directory
config.MODEL.net_structure='MobilenetV3' ######'resnet_v1_50,resnet_v1_101,MobilenetV2
config.MODEL.pretrained_model='./weights/baiguang-mbv3epoch_46L2_5e-05.ckpt'
config.MODEL.task='baiguang'
config.MODEL.min_overlap=0.7
config.MODEL.max_box= 100
config.MODEL.offset= True
config.MODEL.global_stride=4

config.MODEL.deployee= False    ### tensorflow, mnn, coreml
if config.MODEL.deployee:
    config.TRAIN.batch_size = 1
    config.TRAIN.lock_basenet_bn=True
