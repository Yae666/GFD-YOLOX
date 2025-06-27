import torch
import torch.nn as nn
from mmcv.cnn.bricks import Swish

from collections import OrderedDict

from darknet import CSPDarknet
from network_blocks import BaseConv, CSPLayer, DWConv
from yolo_pafpn_asff import ASFF
from HiLoAttention import HiLo
from P2TAttention import PatchEmbed, P2TAttention

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x



class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024, 2048, 4096],
                 depthwise=False, act="silu", epsilon=1e-4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=640, patch_size=16, kernel_size=16, in_chans=3, embed_dim=512,
                                      overlap=False)

        self.p2t_attention = P2TAttention(dim=512, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,
                                          proj_drop=0.,
                                          pool_ratios=[1, 2, 3, 6])

        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.epsilon = epsilon
        self.swish = Swish()
        Conv = DWConv if depthwise else BaseConv


        # 对输入进来的p5进行宽高的下采样 20,20,1024 --> 10, 10, 2048
        self.p5_to_p6 = conv2d(in_channels[2], in_channels[3], kernel_size=3, stride=2)
        # 对p6进行宽高的下采样 10, 10, 2048 --> 5, 5, 4096
        self.p6_to_p7 = conv2d(in_channels[3], in_channels[4], kernel_size=3, stride=2)

        # 简易注意力机制的weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.p6_upsample = Upsample(in_channels[4], in_channels[3])
        self.conv6_up = conv2d(in_channels[3], in_channels[3], kernel_size=3)

        self.p5_upsample = Upsample(in_channels[3], in_channels[2])
        self.conv5_up = conv2d(in_channels[2], in_channels[2], kernel_size=3)

        self.p4_upsample = Upsample(in_channels[2], in_channels[1])
        self.conv4_up = conv2d(in_channels[1], in_channels[1], kernel_size=3)

        self.p3_upsample = Upsample(in_channels[1], in_channels[0])
        self.conv3_up = conv2d(in_channels[0], in_channels[0], kernel_size=3)

        self.p4_downsample = conv2d(in_channels[0], in_channels[1], kernel_size=3, stride=2)
        self.conv4_down = conv2d(in_channels[1], in_channels[1], kernel_size=3)

        self.p5_downsample = conv2d(in_channels[1], in_channels[2], kernel_size=3, stride=2)
        self.conv5_down = conv2d(in_channels[2], in_channels[2], kernel_size=3)

        self.p6_downsample = conv2d(in_channels[2], in_channels[3], kernel_size=3, stride=2)
        self.conv6_down = conv2d(in_channels[3], in_channels[3], kernel_size=3)

        self.p7_downsample = conv2d(in_channels[3], in_channels[4], kernel_size=3, stride=2)
        self.conv7_down = conv2d(in_channels[4], in_channels[4], kernel_size=3)


        # Attention channel size
        self.neck_channels = [256, 512, 1024]

        # HiLo
        # dark5 1024
        self.HiLo_1 = HiLo(dim=1024, num_heads=8, window_size=2, alpha=0.5)
        # dark4 512
        self.HiLo_2 = HiLo(dim=512, num_heads=8, window_size=2, alpha=0.5)
        # dark3 256
        self.HiLo_3 = HiLo(dim=256, num_heads=8, window_size=2, alpha=0.5)

        # PAN CSPLayer pan_out2 downsample 256
        self.HiLo_pa1 = HiLo(dim=int(self.neck_channels[0] * width), num_heads=8, window_size=2, alpha=0.5)
        # PAN CSPLayer pan_out1 downsample 512
        self.HiLo_pa2 = HiLo(dim=int(self.neck_channels[1] * width), num_heads=8, window_size=2, alpha=0.5)
        # PAN CSPLayer pan_out0 1024
        self.HiLo_pa3 = HiLo(dim=int(self.neck_channels[2] * width), num_heads=8, window_size=2, alpha=0.5)


        # ASFF
        self.asff_1 = ASFF(level=0, multiplier=width)    #1024
        self.asff_2 = ASFF(level=1, multiplier=width)
        self.asff_3 = ASFF(level=2, multiplier=width)




    def forward(self, input):
        # print(input.shape)
        x, H, W = self.patch_embed(input)
        # print(x.shape)

        input = self.p2t_attention(x, H, W)
        # print(input.shape)
        input = input.reshape(1, 3, 256, 256)
        # print(input.shape)

        # backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features


        # HiLo
        p5_in = self.HiLo_1(x0)
        p4_in = self.HiLo_2(x1)
        p3_in = self.HiLo_3(x2)


        p6_in = self.p5_to_p6(p5_in)

        p7_in = self.p6_to_p7(p6_in)



        # 简单的注意力机制，用于确定更关注p7_in还是p6_in
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # p6_td 10, 10, 2048
        p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # 简单的注意力机制，用于确定更关注p6_td还是p5_in
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # p5_td 20, 20, 1024
        p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

        # 简单的注意力机制，用于确定更关注p5_td还是p4_in
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # p4_td 40, 40, 512
        p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

        # 简单的注意力机制，用于确定更关注p4_td还是p3_in
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # p3_out 80, 80, 256
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

        # HiLo
        p3_out = self.HiLo_pa1(p3_out)

        # 简单的注意力机制，用于确定更关注p4_in还是p4_td还是p3_out
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # p4_out 40,40,512
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

        # HiLo
        p4_out = self.HiLo_pa2(p4_out)

        # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # p5_out 20, 20, 1024
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

        # HiLo
        p5_out = self.HiLo_pa3(p5_out)

        # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # p6_out 10, 10, 2048
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

        # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # p7_out 5, 5, 4096
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        p5_out = p5_out.reshape(1, 1024, 8, 8)
        p4_out = p4_out.reshape(1, 512, 16, 16)
        p3_out = p3_out.reshape(1, 256, 32, 32)


        outputs = (p3_out, p4_out, p5_out)

        # ASFF
        p5_out = self.asff_1(outputs)
        p4_out = self.asff_2(outputs)
        p3_out = self.asff_3(outputs)
        outputs = (p3_out, p4_out, p5_out)


        return outputs