# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import numpy as np


def select_iou_loss(pred, target, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = pred.size(0)
    assert pred.size(0) == target.size(0)
    target = target.clamp(min=0.)
    area_pred = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    area_gt = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    area_i = ((torch.min(pred[:, 0], target[:, 0]) +
               torch.min(pred[:, 2], target[:, 2])) *
              (torch.min(pred[:, 1], target[:, 1]) +
               torch.min(pred[:, 3], target[:, 3])))
    area_u = area_pred + area_gt - area_i
    iou = area_i / area_u
    loc_losses = -torch.log(iou.clamp(min=1e-7))

    return torch.sum(weight * loc_losses) / avg_factor



class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self._init_layers(num_classes)

    def _init_layers(self, num_classes):
        self.stacked_convs = 3
        self.feat_channels = 256
        self.cls_convs = []
        self.reg_convs = []

        self.cls_convs.append(nn.Conv2d(2048, self.feat_channels, 3, padding=1))
        self.reg_convs.append(nn.Conv2d(2048, self.feat_channels, 3, padding=1))
        self.cls_convs.append(nn.ReLU(inplace=True))
        self.reg_convs.append(nn.ReLU(inplace=True))
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1))
            self.cls_convs.append(nn.ReLU(inplace=True))
            self.reg_convs.append(
                nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1))
            self.reg_convs.append(nn.ReLU(inplace=True))
        self.cls_convs.append(nn.Conv2d(self.feat_channels, num_classes, 3, padding=1))
        self.reg_convs.append(nn.Conv2d(self.feat_channels, 4, 3, padding=1))
        self.reg_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*self.cls_convs)
        self.reg_convs = nn.Sequential(*self.reg_convs)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        anchor_map_cls = self.cls_convs(src)
        anchor_map_box = self.reg_convs(src)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'anchor_point_map': (anchor_map_cls, anchor_map_box, mask)}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def xyxy2xcycwh(self, xyxy):
        """Convert [x1 y1 x2 y2] box format to [xc yc w h] format."""
        return torch.cat(
            (0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]),
            dim=1)

    def xcycwh2xyxy(self, xywh):
        """Convert [xc yc w y] box format to [x1 y1 x2 y2] format."""
        return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4],
                          xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), dim=1)

    def prop_box_bounds(self, boxes, scale, width, height):
        """Compute proportional box regions.

        Box centers are fixed. Box w and h scaled by scale.
        """
        prop_boxes = self.xyxy2xcycwh(boxes)
        prop_boxes[:, 2:] *= scale
        prop_boxes = self.xcycwh2xyxy(prop_boxes)
        x1 = torch.floor(prop_boxes[:, 0]).clamp(0, width - 1).int()
        y1 = torch.floor(prop_boxes[:, 1]).clamp(0, height - 1).int()
        x2 = torch.ceil(prop_boxes[:, 2]).clamp(1, width).int()
        y2 = torch.ceil(prop_boxes[:, 3]).clamp(1, height).int()
        return x1, y1, x2, y2

    def _meshgrid(self, x, y):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        return xx, yy

    def point_target_single(self, cls_score_list, bbox_pred_list, gt_bboxes,
                            gt_labels, s_mask, s_target):
        num_levels = 1
        assert len(cls_score_list) == len(bbox_pred_list) == num_levels
        feat_lvls = torch.zeros_like(gt_labels)

        labels = []
        label_weights = []
        bbox_targets = []
        bbox_locs = []
        device = bbox_pred_list[0].device

        for lvl in range(num_levels):
            stride = 32.
            norm = stride * 4.
            inds = torch.nonzero(feat_lvls == lvl).squeeze(-1)
            h, w = cls_score_list[lvl].size()[-2:]

            _labels = torch.zeros_like(
                cls_score_list[lvl][0], dtype=torch.long)
            _label_weights = 1 - s_mask[lvl].float()
            _bbox_targets = bbox_pred_list[lvl].new_zeros((0, 4),
                                                          dtype=torch.float)
            _bbox_locs = bbox_pred_list[lvl].new_zeros((0, 3),
                                                       dtype=torch.long)
            if len(inds) > 0:
                boxes = gt_bboxes[inds, :]
                classes = gt_labels[inds]
                proj_boxes = boxes / stride
                ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                    proj_boxes, 0.5, w, h)
                pos_x1, pos_y1, pos_x2, pos_y2 = self.prop_box_bounds(
                    proj_boxes, 0.2, w, h)
                for i in range(len(inds)):
                    # setup classification ground-truth
                    _labels[pos_y1[i]:pos_y2[i], pos_x1[i]:
                            pos_x2[i]] = classes[i]
                    _label_weights[ig_y1[i]:ig_y2[i], ig_x1[i]:ig_x2[i]] = 0.
                    _label_weights[pos_y1[i]:pos_y2[i], pos_x1[i]:
                                   pos_x2[i]] = 1.
                    # setup localization ground-truth
                    locs_x = torch.arange(
                        pos_x1[i], pos_x2[i], device=device, dtype=torch.long)
                    locs_y = torch.arange(
                        pos_y1[i], pos_y2[i], device=device, dtype=torch.long)
                    shift_x = (locs_x.float() + 0.5) * stride
                    shift_y = (locs_y.float() + 0.5) * stride
                    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
                    shifts = torch.stack(
                        (shift_xx, shift_yy, shift_xx, shift_yy), dim=-1)
                    shifts[:, 0] = shifts[:, 0] - boxes[i, 0]
                    shifts[:, 1] = shifts[:, 1] - boxes[i, 1]
                    shifts[:, 2] = boxes[i, 2] - shifts[:, 2]
                    shifts[:, 3] = boxes[i, 3] - shifts[:, 3]
                    _bbox_targets = torch.cat((_bbox_targets, shifts / norm),
                                              dim=0)
                    locs_xx, locs_yy = self._meshgrid(locs_x, locs_y)
                    zeros = torch.zeros_like(locs_xx)
                    locs = torch.stack((zeros, locs_yy, locs_xx), dim=-1)
                    _bbox_locs = torch.cat((_bbox_locs, locs), dim=0)

            labels.append(_labels)
            label_weights.append(_label_weights)
            bbox_targets.append(_bbox_targets)
            bbox_locs.append(_bbox_locs)

        # compute number of foreground and background points
        num_pos = 0
        num_neg = 0
        for lvl in range(num_levels):
            npos = bbox_targets[lvl].size(0)
            num_pos += npos
            num_neg += (label_weights[lvl].nonzero().size(0) - npos)
        return (labels, label_weights, bbox_targets, bbox_locs, num_pos,
                num_neg)

    def images_to_levels(self, target, num_imgs, num_levels, is_cls=True):
        level_target = []
        if is_cls:
            for lvl in range(num_levels):
                level_target.append(
                    torch.stack([target[i][lvl] for i in range(num_imgs)],
                                dim=0))
        else:
            for lvl in range(num_levels):
                level_target.append(
                    torch.cat([target[j][lvl] for j in range(num_imgs)],
                              dim=0))
        return level_target

    def mmdet_sigmoid_focal_loss(self, pred, target, weight, gamma, alpha, avg_factor):
        pred_sigmoid = pred.sigmoid()
        targ = torch.zeros_like(pred_sigmoid)
        ind = (target > 0).nonzero().squeeze(-1)
        for i in ind:
            label = target[i] - 1
            targ[i][label] = 1
        target = targ.type_as(pred)

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        loss = loss * weight.unsqueeze(-1)
        loss = loss.sum() / avg_factor
        return loss

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_locs, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, 91)
        loss_cls = self.mmdet_sigmoid_focal_loss(
            cls_score,
            labels,
            weight=label_weights,
            gamma=2,
            alpha=0.25,
            avg_factor=num_total_samples)
        # localization loss
        if bbox_targets.size(0) == 0:
            loss_bbox = bbox_pred.new_zeros(1)
        else:
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred[bbox_locs[:, 0], bbox_locs[:, 1],
                                  bbox_locs[:, 2], :]
            loss_bbox = select_iou_loss(
                bbox_pred,
                bbox_targets,
                1.,
                avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss_anchor_point(self, outputs, targets, indices, num_boxes):
        losses = {}
        anchor_map_cls = outputs['anchor_point_map'][0]  # 2x92x24x32
        anchor_map_box = outputs['anchor_point_map'][1]  # 2x4x24x32
        anchor_map_mask = outputs['anchor_point_map'][2]  # 2x4x24x32

        assert len(targets) == len(anchor_map_cls) == len(anchor_map_box)
        cls_scores = [anchor_map_cls]
        bbox_preds = [anchor_map_box]
        masks = [anchor_map_mask]
        gt_bboxes = []
        gt_labels = []
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            this_size = target['size']
            w, h = this_size[0], this_size[1]
            decoder_boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
            decoder_boxes = box_ops.box_cxcywh_to_xyxy(decoder_boxes)
            gt_bboxes.append(decoder_boxes)
            gt_labels.append(labels)

        # point target
        num_imgs = len(gt_bboxes)
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]

        if gt_labels is None:
            gt_labels = [None for _ in range(num_imgs)]

        num_levels = 1
        assert len(cls_scores) == len(bbox_preds) == num_levels
        cls_score_list = []
        bbox_pred_list = []
        mask_list = []
        for img_id in range(num_imgs):
            cls_score_list.append(
                [cls_scores[i][img_id].detach() for i in range(num_levels)])
            bbox_pred_list.append(
                [bbox_preds[i][img_id].detach() for i in range(num_levels)])
            mask_list.append(
                [masks[i][img_id].detach() for i in range(num_levels)])

        all_labels, all_label_weights, all_bbox_targets, all_bbox_locs, num_pos_list, num_neg_list = [],[],[],[],[],[]
        # point target single
        for s_cls_score_list, s_bbox_pred_list, s_gt_labels, s_gt_bboxes, s_mask, s_target in zip(cls_score_list, bbox_pred_list, gt_labels, gt_bboxes, mask_list, targets):
            labels, label_weights, bbox_targets, bbox_locs, num_pos, num_neg = self.point_target_single(
                s_cls_score_list, s_bbox_pred_list, s_gt_bboxes, s_gt_labels, s_mask, s_target)
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_locs.append(bbox_locs)
            num_pos_list.append(num_pos)
            num_neg_list.append(num_neg)

        for i in range(num_imgs):
            for lvl in range(num_levels):
                all_bbox_locs[i][lvl][:, 0] = i

        # sampled points of all images
        num_total_pos = sum([max(num, 1) for num in num_pos_list])
        num_total_neg = sum([max(num, 1) for num in num_neg_list])
        # combine targets to a list w.r.t. multiple levels
        labels_list = self.images_to_levels(all_labels, num_imgs, num_levels,
                                            True)
        label_weights_list = self.images_to_levels(all_label_weights, num_imgs,
                                                   num_levels, True)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_imgs,
                                                  num_levels, False)
        bbox_locs_list = self.images_to_levels(all_bbox_locs, num_imgs,
                                               num_levels, False)

        # end point target
        num_total_samples = num_total_pos
        # loss single
        # only one feature level so only one loss single.  TODO: for loop for multi level
        loss_cls, loss_bbox = self.loss_single(cls_scores[0], bbox_preds[0], labels_list[0], label_weights_list[0],
                                               bbox_targets_list[0], bbox_locs_list[0], num_total_samples)

        losses['anchor_map_cls'] = loss_cls
        losses['anchor_map_box'] = loss_bbox

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'anchor_point': self.loss_anchor_point,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'anchor_point':
                        #  no need to do anchor point loss here
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.anchor_point:
        losses += ["anchor_point"]
        weight_dict['anchor_map_cls'] = 1.
        weight_dict['anchor_map_box'] = 1.
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
