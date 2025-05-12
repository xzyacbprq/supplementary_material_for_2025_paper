# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as FF
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..utils import resize


    
@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head_1: OptConfigType = None,
                 auxiliary_head_2: OptConfigType = None,
                 auxiliary_head_3: OptConfigType = None,
                 num_class: int = 5,
                 feat_channel: list = [256, 512, 1024, 2048],
                 use_aux_head: str = '',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        
        if len(use_aux_head):
            #self.auxiliary_heads  =  [MODELS.build(auxiliary_head_1)]
            self.auxiliary_heads  =  [MODELS.build(auxiliary_head_2)]
            self.auxiliary_heads +=  [MODELS.build(auxiliary_head_3)]
            _nc = sum([_head.num_classes for _head in self.auxiliary_heads])
    
        self.train_cfg = train_cfg
        self.test_cfg  = test_cfg
        if use_aux_head == 'dlv3':
            self.resize_ind  = 1
            self.resize_frac = 2
            out_ch         = max(feat_channel[0]//self.resize_frac, 32)
            self.conv1     = nn.Conv2d(feat_channel[0],out_ch,1)
            self.conv2     = nn.Conv2d(feat_channel[1],out_ch,1)
            self.conv3     = nn.Conv2d(feat_channel[2],out_ch,1)
            self.conv4     = nn.Conv2d(feat_channel[3],out_ch,1)
            ch = _nc + 4*out_ch
            self.sp_attn   = nn.Conv2d(ch, 1 ,1 )
            
        if use_aux_head in ['upn', 'sfm']:
            self.resize_ind  = 0
            self.resize_frac = 2
            out_ch         = max(feat_channel[0]//self.resize_frac, 256)
            self.conv1     = nn.Conv2d(feat_channel[0],out_ch,1)
            self.conv2     = nn.Conv2d(feat_channel[1],out_ch,1)
            self.conv3     = nn.Conv2d(feat_channel[2],out_ch,1)
            self.conv4     = nn.Conv2d(feat_channel[3],out_ch,1)
            ch = _nc + 4*out_ch
            self.sp_attn   = nn.Conv2d(ch, 1 ,1 )
        else:
            self.resize_ind  = 0
            self.resize_frac = 1
        
        #self.ch_attn   = nn.Linear(ch, ch, bias = False)
        self.use_aux_head = use_aux_head
        assert self.with_decode_head
        
    def resize(self, inp , size):
        return resize(inp, size, mode = 'bilinear', align_corners = self.align_corners)
    
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def extract_feat(self, inputs: Tensor, data_samples : SampleList = None) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        aux_x = []
        y_atn = []
        loss_aux = {}
        if len(self.use_aux_head):
            aux_x, loss_aux = self._auxiliary_head_forward(x, data_samples)
            _size  = x[self.resize_ind].shape[2:]
            
            _x0 = self.conv1(x[0])
            _x0 = self.resize(_x0, _size)
            _x1 = self.conv2(x[1])
            _x1 = self.resize(_x1, _size)
            _x2 = self.conv3(x[2])
            _x2 = self.resize(_x2, _size)
            _x3 = self.conv4(x[3])
            _x3 = self.resize(_x3, _size)
            
            _aux_x = [None,None]
            #_aux_x[0] = self.resize(aux_x[0], _size)
            _aux_x[0] = self.resize(aux_x[0], _size)
            _aux_x[1] = self.resize(aux_x[1], _size)
            
            y   = cat([_x0, _x1, _x2, _x3, _aux_x[0], _aux_x[1]], dim =1)
            #y_aux = cat([aux_x[1], aux_x[2]],dim = 1)
            y_atn = self.sp_attn(y)
            y_atn = F.sigmoid(y_atn)
            if self.use_aux_head == 'dlv3': 
                x = [x[0], y * y_atn]
            elif self.use_aux_head in ['upn', 'sfm']:
                x = [self.resize(cat([_x0, _aux_x[0]], dim = 1) * y_atn, x[0].shape[2:]),
                     self.resize(cat([_x1, _aux_x[1]], dim = 1) * y_atn, x[1].shape[2:]),
                     self.resize(_x2 * y_atn, x[2].shape[2:]),
                     self.resize(_x3 * y_atn, x[3].shape[2:])]
            else:
                print('Auxiliary head is not valid')
        return x, aux_x, y_atn, loss_aux

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, _, _, _ = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        return seg_logits

    def _decode_head_forward(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        feat = self.decode_head.forward(inputs)
        if data_samples != None:
            loss_decode = self.decode_head.loss_by_feat(feat, data_samples)
            losses.update(add_prefix(loss_decode, 'decode'))
        return feat, losses
        
    def _auxiliary_head_forward(self, inputs: List[Tensor],
                                      data_samples: SampleList = None) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        '''
        if isinstance(self.auxiliary_head_1, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
        '''
        aux_outs = []
        for i,head in enumerate(self.auxiliary_heads):
            feat = head.forward(inputs)
            aux_outs.append(feat)
            if data_samples != None:
                loss_aux = head.loss_by_feat(feat, data_samples)
                losses.update(add_prefix(loss_aux, f'aux_{i}'))

        return aux_outs, losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x, _, _, loss_aux = self.extract_feat(inputs, data_samples)
        losses = dict()    
        _, loss_decode  = self._decode_head_forward(x, data_samples)
        losses.update(loss_decode)
        losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_img_metas = [
            dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
        ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        data_samples['pred_sem_seg'] = seg_logits.argmax(dim=1, keepdim=True)
        return  data_samples#self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x, _, _, _ = self.extract_feat(inputs)
        seg_logits = self.decode_head.forward(x)
        seg_logits = seg_logits.argmax(dim=1, keepdim=True)
        if seg_logits.shape[-1] != inputs.shape[-1]:
            seg_logits = FF.resize(seg_logits, inputs.shape[-2:], interpolation = FF.InterpolationMode.BILINEAR)
        data_samples['pred_sem_seg'] = seg_logits
        return data_samples

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentatOne thing I am feeling I should share with everyone as a Muslim. As Muslims we are not being able to portray our proper images in various social aspects now a days. There is no place in Islam for a Mob to take any law in their own hand. It is a unanimously understood matter in Islam. In my understanding, the best you can do in this country is as follows - [i] go to the court of the country if it can help [ii] conduct a peaceful protest [iii] start an awareness campaign showcasing reference and evidence against what you find incorrect religiously. Then it is up to the people whether they take it or leave it. Allah will not even ask his prophets (Peace be Upon Them) “Why you could not make people obey the truth ?”, instead He will ask - “Did you tell people the truth ?”. It doesn’t matter how religious you are, you don’t have more responsibility than the Prophets (Peace be Upon Them) of Allah.

  

ions.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
