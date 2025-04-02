import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP, SPP


class S2PM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out
    
class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)
        
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        m_batchsize1, C1, height1, width1 = guiding_map0.size()
     

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class FPNNeck(nn.Module):
    def __init__(self, inplanes, num_class):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)

        self.decoder = nn.Sequential(BasicConv2d(inplanes*8, inplanes, 3, 1, 1), nn.Conv2d(inplanes, 1, 3, 1, 1))
        

        self.cgm_2 = ChangeGuideModule(inplanes*4)
        self.cgm_3 = ChangeGuideModule(inplanes*8)
        self.cgm_4 = ChangeGuideModule(inplanes*8)

        # 相比v2 额外的模块
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(inplanes*4, inplanes*4, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(inplanes*2, inplanes*2, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(inplanes*1, inplanes*1, 3, 1, 1)

        # self.decoder_final = nn.Sequential(BasicConv2d(inplanes*1, inplanes, 3, 1, 1), nn.Conv2d(inplanes, 2, 1))
        
        # self.fcn_out = nn.Sequential(Conv3Relu(2, 2),
        #                           nn.Dropout(0.2),  
        #                           nn.Conv2d(2, num_class, (1, 1)))
        
        self.seg_decoder1 = S2PM(2 * inplanes, 1 *inplanes)
        self.seg_decoder2 = S2PM(4 * inplanes, 2 *inplanes)
        self.seg_decoder3 = S2PM(8 * inplanes, 4 *inplanes)
        self.SIM1 = SIM(norm_nc=inplanes, label_nc=inplanes, nhidden=inplanes)
        self.SIM2 = SIM(norm_nc=2 *inplanes, label_nc=2 *inplanes, nhidden=2 *inplanes)
        self.SIM3 = SIM(norm_nc=4 *inplanes, label_nc=4 *inplanes, nhidden=4 *inplanes)


    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        # change1 = self.stage1_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 2
        # change2 = self.stage2_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        # change3 = self.stage3_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 8 
        # change4 = self.stage4_Conv1(torch.cat([fa1, fb1], 1))  # inplanes * 8

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8

        change4_1 = F.interpolate(change4, change1.size()[2:], mode='bilinear', align_corners=True)
        #feature_fuse = change4_1 

        #change_map = self.decoder(feature_fuse) 
        change4 = self.seg_decoder3(change4_1)
        feature4 = self.decoder_module4(self.SIM3(change4, change3))
        change3 = self.seg_decoder2(feature4)
        feature3 = self.decoder_module3(self.SIM2(change3, change2))
        change2 = self.seg_decoder1(feature3)
        change1 = self.decoder_module2(self.SIM1(change2, change1))

        #change_map = F.interpolate(change_map, (256,256), mode='bilinear', align_corners=True)

        # final_map = self.decoder_final(change1)``
        # final_map = self.fcn_out(final_map)
        # final_map = F.interpolate(final_map,  (256,256), mode='bilinear', align_corners=True)

        return change1
        

     
