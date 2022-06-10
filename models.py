import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.vgg import vgg16_bn, vgg16
# from resnet_no_relu import resnet18_nr
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, pretrained=False, bn=False):
        super(VGG16, self).__init__()
        if bn:
            model = vgg16_bn(pretrained=pretrained)
            self.modules = list(model.features)
            self.block0 = nn.Sequential(*self.modules[0:17])
            self.block1 = nn.Sequential(*self.modules[17:23])
            self.block2 = nn.Sequential(*self.modules[23:33])
            self.block3 = nn.Sequential(*self.modules[33:43])
        else:
            model = vgg16(pretrained=pretrained)
            self.modules = list(model.features)
            self.block0 = nn.Sequential(*self.modules[0:12])
            self.block1 = nn.Sequential(*self.modules[12:16])
            self.block2 = nn.Sequential(*self.modules[16:23])
            self.block3 = nn.Sequential(*self.modules[23:30])

    def forward(self, x):
        # 64x64x256
        out0 = self.block0(x)
        # 64x64x256
        out1 = self.block1(out0)
        # 32x32x512
        out2 = self.block2(out1)
        # 16x16x512
        out3 = self.block3(out2)
        return {"out2": out0,
                "out3": out1,
                "out4": out2,
                "out5": out3
                }


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        model = resnet18(pretrained=pretrained)

        modules = list(model.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = modules[7]

    def forward(self, x):
        x = self.block1(x)
        # 64x64x64
        out2 = self.block2(x)
        # 32x32x128
        out3 = self.block3(out2)
        # 16x16x256
        out4 = self.block4(out3)
        # 8x8x512
        out5 = self.block5(out4)
        return {"out2": out2,
                "out3": out3,
                "out4": out4,
                "out5": out5
                }


# import copy
# class ResNet18NR(nn.Module):
#     def __init__(self, pretrained=False):
#         super(ResNet18NR, self).__init__()
#         model_nr = resnet18_nr(pretrained=False)
#         modules_nr = list(model_nr.children())


#         self.block2_logvar = copy.copy(nn.Sequential(*modules_nr[0:5]))
#         self.block3_logvar = copy.copy(nn.Sequential(modules_nr[5]))
#         self.block4_logvar = copy.copy(nn.Sequential(modules_nr[6]))

#     def forward(self, x):
#         out2_logvar = self.block2_logvar(x)
#         out3_logvar = self.block3_logvar(out2_logvar)
#         out4_logvar = self.block4_logvar(out3_logvar)
#         # 8x8x512
#         return {"out2": out2_logvar,
#                 "out3": out3_logvar,
#                 "out4": out4_logvar
#                 }



class ResNet34(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34, self).__init__()
        model = resnet34(pretrained=pretrained)

        modules = list(model.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = modules[7]

    def forward(self, x):
        x = self.block1(x)
        # 64x64x64
        out2 = self.block2(x)
        # 32x32x128
        out3 = self.block3(out2)
        # 16x16x256
        out4 = self.block4(out3)
        # 8x8x512
        out5 = self.block5(out4)
        return {"out2": out2,
                "out3": out3,
                "out4": out4,
                "out5": out5
                }


class KDLoss(nn.Module):
    def __init__(self, loss_type=None):
        super(KDLoss, self).__init__()
        '''
        loss type: l2, l1, consine, l2+consine, l2norm+l2
        '''
        self.type = loss_type

    def get_loss_map(self, feat_T, feat_S):
        '''
        :param feat_T: NxCxHxW
        :param feat_S: NxCxHxW
        :return:
        '''
        if self.type == "l2norm+l2":
            feat_T = F.normalize(feat_T, p=2, dim=1)
            feat_S = F.normalize(feat_S, p=2, dim=1)

            loss_map = 0.5 * ((feat_T - feat_S) ** 2)
            loss_map = torch.sum(loss_map, dim=1)

        elif self.type == "l1norm+l2":
            feat_T = F.normalize(feat_T, p=1, dim=1)
            feat_S = F.normalize(feat_S, p=1, dim=1)

            loss_map = 0.5 * ((feat_T - feat_S) ** 2)
            loss_map = torch.sum(loss_map, dim=1)

        elif self.type == "consine":
            feat_T = F.normalize(feat_T, p=2, dim=1)
            feat_S = F.normalize(feat_S, p=2, dim=1)
            loss_map = 1 - torch.sum(torch.mul(feat_T, feat_S), dim=1)

        elif self.type == "l2":
            loss_map = (feat_T - feat_S) ** 2
            loss_map = torch.sum(loss_map, dim=1)

        elif self.type == "l1":
            loss_map = torch.abs(feat_T - feat_S)
            loss_map = torch.sum(loss_map, dim=1)

        else:
            raise NotImplementedError

        return loss_map

    def forward(self, feat_T, feat_S):
        loss_map = self.get_loss_map(feat_T, feat_S)
        # if self.type == "consine":
        #     return torch.mean(loss_map)
        # else:
        #     return torch.sum(loss_map)
        return torch.sum(torch.mean(loss_map, dim=(1, 2)))
        # if use mean, must increase the learning rate
        # return torch.mean(loss_map)


class Conv_BN_PRelu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, prelu=True):
        super(Conv_BN_PRelu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if prelu:
            self.conv.append(nn.PReLU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class NonLocalAttention(nn.Module):
    def __init__(self, channel=256, reduction=2, rescale=1.0):
        super(NonLocalAttention, self).__init__()
        # self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
         
        self.conv_match1 = Conv_BN_PRelu(channel, channel//reduction, 1, bn=False, prelu=True)
        self.conv_match2 = Conv_BN_PRelu(channel, channel//reduction, 1, bn=False, prelu=True)
        self.conv_assembly = Conv_BN_PRelu(channel, channel, 1,bn=False, prelu=True)
        self.rescale = rescale

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        x_final = x_final.permute(0,2,1).view(N,-1,H,W)
        return x_final + input*self.rescale
        # return x_final 


if __name__ == "__main__":
    x = torch.rand([2, 3, 256, 256])
    T = ResNet18NR(True)
    T(x)
    S = ResNet18NR(False)
