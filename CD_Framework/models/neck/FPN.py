import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import PPM, ASPP, SPP

class FPNNeck(nn.Module):
    # 初始参数绑定,常改通道数
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()

        # Conv3Relu <---> models.block中的Base.py(我们框架写用的卷积模块)
    
        # "一"中的特征融合
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel:  2*inplanes ---> inplanes # 输入backbone后给的通道数*2,输出backbone后给的通道数不变
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel:  4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel:  8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel:  16*inplanes ---> 8*inplanes

        # "二"中的降维变大
        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes) # channel:  2*inplanes ---> inplanes
        # 对change2进行通道变少,尺寸变大的函数(降维变大) ---> 为了与change1融合
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2) # channel:  4*inplanes ---> 2*inplanes
        # 对change3进行通道变少,尺寸变大的函数(降维变大) ---> 为了与change2融合
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4) # channel:  8*inplanes ---> 4*inplanes
        # 对change4进行通道变少,尺寸变大的函数(降维变大) ---> 为了与change3融合


        # 对change1,2,3进行多尺度融合后进行的
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        # PPM/ASPP比SPP好
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        # "三"中的降维变大(通道变少,尺寸变大)
        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)

            self.fuse = True
        else:
            self.fuse = False

        # 上采样实例化与其他实例化
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.d2 = Conv3Relu(inplanes * 2, inplanes)
        self.d3 = Conv3Relu(inplanes * 4, inplanes)
        self.d4 = Conv3Relu(inplanes * 8, inplanes)

        # 正则化（惩罚项，避免过拟合）<---> 提升鲁棒性 <---> 框架FPN的一大新意 <---> 放在有效的位置可实现好效果
        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        # 融合分一、二、三步进行;其中,在第二、三步中均遇到降维变大的融合手段
        # 降维变大手段 : 二:"上采样+卷积" 三:"插值+卷积"

        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        # inplanes *1 *2 *4 *8 
        # torch.Size([32, 96, 56, 56]) torch.Size([32, 192, 28, 28]) torch.Size([32, 384, 14, 14]) torch.Size([32, 768, 7, 7])
        # torch.Size([32, 96, 56, 56]) torch.Size([32, 192, 28, 28]) torch.Size([32, 384, 14, 14]) torch.Size([32, 768, 7, 7])
        change1_h, change1_w = fa1.size(2), fa1.size(3)
        # print(fa1.shape,fa2.shape,fa3.shape,fa4.shape)
        # print(fb1.shape,fb2.shape,fb3.shape,fb4.shape)


        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock:正则化（惩罚项，避免过拟合）<---> 提升鲁棒性 <---> 框架一大新意

        ## 一.两个时序"同尺寸特征图"的融合阶段

        # 首先对8张特征图 进行 分四个阶段进行"不同时序 同尺寸"的特征融合 # print(change1.shape,change2.shape,change3.shape,change4.shape)
        # 从1到4,尺寸越来越小，通道越来越多 ; 最小的图为高维特征 
        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes = inplanes
        # 纯净torch.Size([32, 96, 56, 56])
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes = inplanes * 2
        # 纯净torch.Size([32, 192, 28, 28])
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes = inplanes * 4
        # 纯净torch.Size([32, 384, 14, 14])
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes = inplanes * 8
        # 纯净torch.Size([32, 768, 7, 7])

        if self.expand_field is not None:  # 扩大感受野
            change4 = self.expand_field(change4)
        # change4选择models.block中Field.py的三个感受野扩大方式 ---> 选择的 aspp

        ## 二.多尺度渐进式"增强的特征融合"阶段 ---> 增强后的特征

        change3_2 = self.stage4_Conv_after_up(self.up(change4)) # print(change3_2.shape)
        # "上采样+卷积" <--> 减少change4的通道,放大change4的尺寸 ---> 变成change3_2 ---> 为了与change3融合
        # torch.Size([32, 384, 14, 14]) 使变成change3的张量通道及尺寸
        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1))
        # 融合后对change3再进行一次卷积(特征增强)
        # torch.Size([32, 384, 14, 14])
        # 得到融合卷积后的新change3


        change2_2 = self.stage3_Conv_after_up(self.up(change3)) # print(change2_2.shape)
        # "上采样+卷积" <--> 减少"新change3"的通道,放大change3的尺寸 ---> 变成change2_2 ---> 为了与change2融合
        # torch.Size([32, 192, 28, 28]) 使变成change2的张量通道及尺寸
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1))
        # 融合后对change2再进行一次卷积(特征增强)
        # torch.Size([32, 192, 28, 28])
        # 融合卷积后的新change2


        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        # "上采样+卷积" <--> 减少"新change2"的通道,放大change2的尺寸 ---> 变成change1_2 ---> 为了与change1融合
        # torch.Size([32, 96, 56, 56]) 使变成change1的张量通道及尺寸
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1))
        # 融合后对change1再进行一次卷积(特征增强)
        # torch.Size([32, 96, 56, 56])
        # 融合后的change1


        ## 三. 降维融合阶段 ---> 全部转换为change1的尺寸大小(降维变大)进行最终阶段的融合

        if self.fuse:
            # "插值+卷积" <--> 对change2,3,4四个特征图进行"降维变大"
            # interpolate : 对特征图进行插值
            change4 = self.stage4_Conv3(F.interpolate(change4, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))            
            change3 = self.stage3_Conv3(F.interpolate(change3, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))
            change2 = self.stage2_Conv3(F.interpolate(change2, size=(change1_h, change1_w),
                                                      mode='bilinear', align_corners=True))

            [change1, change2, change3, change4] = self.drop([change1, change2, change3, change4])  # dropblock:正则化（惩罚项，避免过拟合）<---> 提升鲁棒性 <---> 框架一大新意

            change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1)) #  最终的融合卷积
        else:
            change = change1

        # change2 = self.d2(self.up(change2))
        # change3 = self.d3(self.up4(change3))
        # change4 = self.d4(self.up8(change4))
        # change = self.final_Conv(torch.cat([change1, change2, change3, change4], 1))

        return change

