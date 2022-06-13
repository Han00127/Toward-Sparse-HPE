import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH

import utils


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)
        # utils.compute_rotation_matrix_from_ortho6d(x)


class HPEHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(HPEHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 6, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 6)


class RetinaFace_HPE(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace_HPE, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.HPEHead = self._make_hpe_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def _make_hpe_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        hpehead = nn.ModuleList()
        for i in range(fpn_num):
            hpehead.append(HPEHead(inchannels, anchor_num))
        return hpehead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        hpe_regressions = torch.cat([self.HPEHead[i](feature) for i, feature in enumerate(features)], dim=1)
        hpe_regressions = compute_rotation_matrix_from_ortho6d(hpe_regressions)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions, hpe_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions, hpe_regressions)
        return output

#poses batch*a*6
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):
    b, a, _ = poses.shape
    x_raw = poses[:, :, 0:3]  # batch*a*3
    y_raw = poses[:, :, 3:6]  # batch*a*3

    x = normalize_vector(x_raw, use_gpu)  # batch*a*3
    z = cross_product(x, y_raw)  # batch*a*3
    z = normalize_vector(z, use_gpu)  # batch*a*3
    y = cross_product(z, x)  # batch*a*3

    x = x.view(b, a, 3, 1)
    y = y.view(b, a, 3, 1)
    z = z.view(b, a, 3, 1)
    matrix = torch.cat((x, y, z), -1)  # batch*a*3*3
    return matrix


# batch*a*n
def normalize_vector(v, use_gpu=True):
    batch, a = v.shape[:-1]
    v_mag = torch.sqrt(v.pow(2).sum(2))  # batch*a
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch, a, 1).expand(batch, a, v.shape[-1])
    v = v / v_mag
    return v


# u, v batch*a*n
def cross_product(u, v):
    batch, a = u.shape[:-1]
    # print (u.shape)
    # print (v.shape)
    i = u[:, :, 1] * v[:, :, 2] - u[:, :, 2] * v[:, :, 1]
    j = u[:, :, 2] * v[:, :, 0] - u[:, :, 0] * v[:, :, 2]
    k = u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]

    out = torch.cat((i.view(batch, a, 1), j.view(batch, a, 1), k.view(batch, a, 1)), -1)  # batch*a*3

    return out
