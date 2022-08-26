# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
import paddle
import numpy as np

from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def __init__(self, backbone=None, head=None):
        super().__init__(backbone=backbone, head=head)
        self.avgpool2d = paddle.nn.AdaptiveAvgPool2D((1, 1), data_format='NCHW')

    def forward_net(self, imgs):
        # NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase, should obtain it from imgs(paddle.Tensor) now, then call self.head method.
        num_segs = imgs.shape[
            1]  # imgs.shape=[N,T,C,H,W], for most commonly case
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))

        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = imgs
        #补充加入的，使得每一帧提取出的特征由【batch*frames，2048，7，7】-》 【batch*frames，2048】
        feature = self.avgpool2d(feature)
        x = paddle.reshape(feature, [-1, num_segs, feature.shape[1]])
        imgs=[(x,paddle.to_tensor([num_segs]*x.shape[0]),paddle.ones_like(x))]
        if self.head is not None:
            # print("feature==============================================>>>>>>", feature)
            # print("num_segs==============================================>>>>>>", num_segs)
            cls_score = self.head(imgs)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        imgs = data_batch[0]
        labels = np.array(data_batch[1:])
        batch_size=labels.shape[1]
        labels=paddle.reshape(paddle.to_tensor(labels),[batch_size,-1])
        cls_score = self.forward_net(imgs)
        loss = self.head.loss(cls_score,labels)
        top1,top5 = self.head.metric(cls_score, labels)
        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['top1'] = top1
        loss_metrics['top5'] = top5

        return loss_metrics

    def val_step(self, data_batch):
        imgs = data_batch[0]
        labels = np.array(data_batch[1:])
        batch_size=labels.shape[1]
        labels=paddle.reshape(paddle.to_tensor(labels),[batch_size,-1])
        cls_score = self.forward_net(imgs)
        loss = self.head.loss(cls_score,labels)
        top1,top5 = self.head.metric(cls_score, labels)
        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['top1'] = top1
        loss_metrics['top5'] = top5

        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss, we deal with the test processing in /paddlevideo/metrics
        imgs = data_batch[0]
        cls_score = self.forward_net(imgs)
        return cls_score

    # def infer_step(self, data_batch):
    #     """Define how the model is going to test, from input to output."""
    #     imgs = data_batch[0]
    #     cls_score = self.forward_net(imgs)
    #     return cls_score
    
    def infer_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        imgs = data_batch[0]
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))
        feature = self.backbone(imgs)
        feat = self.avgpool2d(feature)
        return feat
