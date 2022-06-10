'''
    Unsupervised Image Anomaly Detection and Segmentation Based on Pre-trained Feature Mapping
'''
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.vgg import vgg16_bn, vgg19_bn
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import visualize, cal_pro_metric_new, set_seed
from loguru import logger
import argparse


class PretrainedModel(nn.Module):
    def __init__(self, model_name):
        super(PretrainedModel, self).__init__()
        if "resnet" in model_name:
            model = eval(model_name)(pretrained=True)
            modules = list(model.children())
            self.block1 = nn.Sequential(*modules[0:4])
            self.block2 = modules[4]
            self.block3 = modules[5]
            self.block4 = modules[6]
            self.block5 = modules[7]
        elif "vgg" in model_name:
            model = eval(model_name)(pretrained=True)
            self.modules = list(model.features)
            if model_name == "vgg16_bn":
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:23])
                self.block3 = nn.Sequential(*self.modules[23:33])
                self.block4 = nn.Sequential(*self.modules[33:43])
            else:
                self.block1 = nn.Sequential(*self.modules[0:14])
                self.block2 = nn.Sequential(*self.modules[14:26])
                self.block3 = nn.Sequential(*self.modules[26:39])
                self.block4 = nn.Sequential(*self.modules[39:52])

        else:
            raise NotImplementedError

    def forward(self, x):
        # B x 64 x 64 x 64
        out1 = self.block1(x)
        # B x 128 x 32 x 32
        out2 = self.block2(out1)
        # B x 256 x 16 x 16
        # 32x32x128
        out3 = self.block3(out2)
        # 16x16x256
        out4 = self.block4(out3)
        return {"out2": out2,
                "out3": out3,
                "out4": out4
                }

# x = torch.rand([1, 3, 256, 256])
# # T = ResNetS(model_name='resnet18', pretrained=True)
# T = PretrainedModel(model_name='vgg16_bn')
# y = T(x)
#
# for key in y.keys():
#     print(f"{key}: {y[key].shape}")

class Conv_BN_Relu(nn.Module):
    def __init__(self, in_dim, out_dim, k=1, s=1, p=0, bn=True, relu=True):
        super(Conv_BN_Relu, self).__init__()
        self.conv = [
            nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p),
        ]
        if bn:
            self.conv.append(nn.BatchNorm2d(out_dim))
        if relu:
            self.conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)



class DualProjectionNet(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, latent_dim=256):
        super(DualProjectionNet, self).__init__()
        self.encoder1 = nn.Sequential(*[
            Conv_BN_Relu(in_dim, in_dim//2+latent_dim),
            Conv_BN_Relu(in_dim//2+latent_dim, 2*latent_dim),
            # Conv_BN_Relu(2*latent_dim, latent_dim),
        ])

        self.shared_coder = Conv_BN_Relu(2*latent_dim, latent_dim, bn=False, relu=False)

        self.decoder1 = nn.Sequential(*[
            Conv_BN_Relu(latent_dim, 2*latent_dim),
            Conv_BN_Relu(2*latent_dim, out_dim//2+latent_dim),
            Conv_BN_Relu(out_dim//2+latent_dim, out_dim, bn=False, relu=False),
        ])


        self.encoder2 = nn.Sequential(*[
            Conv_BN_Relu(out_dim, out_dim // 2 + latent_dim),
            Conv_BN_Relu(out_dim // 2 + latent_dim, 2 * latent_dim),
            # Conv_BN_Relu(2 * latent_dim, latent_dim),
        ])

        self.decoder2 = nn.Sequential(*[
            Conv_BN_Relu(latent_dim, 2 * latent_dim),
            Conv_BN_Relu(2 * latent_dim, in_dim // 2 + latent_dim),
            Conv_BN_Relu(in_dim // 2 + latent_dim, in_dim, bn=False, relu=False),
        ])


    def forward(self, xs, xt):
        xt_hat = self.encoder1(xs)
        xt_hat = self.shared_coder(xt_hat)
        xt_hat = self.decoder1(xt_hat)

        xs_hat = self.encoder2(xt)
        xs_hat = self.shared_coder(xs_hat)
        xs_hat = self.decoder2(xs_hat)

        return xs_hat, xt_hat


class DFP_AD(object):
    def __init__(self, type='vgg'):
        if type == "resnet":
            self.Agent1 = PretrainedModel(model_name="resnet18")
            self.Agent2 = PretrainedModel(model_name="resnet34")

        elif type == "vgg":
            self.Agent1 = PretrainedModel(model_name="vgg16_bn")
            self.Agent2 = PretrainedModel(model_name="vgg19_bn")

    def register(self, **kwargs):
        self.class_name = kwargs['class_name']
        self.device = kwargs['device']
        self.trainloader = kwargs['trainloader']
        self.testloader = kwargs['testloader']


        self.projector2 = DualProjectionNet(in_dim=256, out_dim=256, latent_dim=200)
        self.optimizer2 = torch.optim.Adam(self.projector2.parameters(), lr=kwargs["lr2"], weight_decay=kwargs["weight_decay"])
        self.projector3 = DualProjectionNet(in_dim=512, out_dim=512, latent_dim=400)
        self.optimizer3 = torch.optim.Adam(self.projector3.parameters(), lr=kwargs["lr3"], weight_decay=kwargs["weight_decay"])
        self.projector4 = DualProjectionNet(in_dim=512, out_dim=512, latent_dim=400)
        self.optimizer4 = torch.optim.Adam(self.projector4.parameters(), lr=kwargs["lr4"], weight_decay=kwargs["weight_decay"])

        self.Agent1.to(self.device).eval()
        self.Agent2.to(self.device).eval()
        self.projector2.to(self.device)
        self.projector3.to(self.device)
        self.projector4.to(self.device)

        self.save_root = "./result/MB-PFM-VGG_{}/".format(kwargs["seed"])
        os.makedirs(os.path.join(self.save_root, "ckpt"), exist_ok=True)
        self.ckpt2 = os.path.join(self.save_root, "ckpt/{}_2.pth".format(kwargs["class_name"]))
        self.ckpt3 = os.path.join(self.save_root, "ckpt/{}_3.pth".format(kwargs["class_name"]))
        self.ckpt4 = os.path.join(self.save_root, "ckpt/{}_4.pth".format(kwargs["class_name"]))
        os.makedirs(os.path.join(self.save_root, "tblogs"), exist_ok=True)
        self.tblog = os.path.join(self.save_root, "tblogs/{}".format(kwargs["class_name"]))


    def get_agent_out(self, x):
        out_a1 = self.Agent1(x)
        out_a2 = self.Agent2(x)
        for key in out_a2.keys():
            out_a1[key] = F.normalize(out_a1[key], p=2)
            out_a2[key] = F.normalize(out_a2[key], p=2)
        return out_a1, out_a2

    def train(self, epochs=100):
        if not os.path.exists(self.ckpt2):
            if os.path.exists(self.tblog):
                shutil.rmtree(self.tblog)
            os.makedirs(self.tblog, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tblog)
            for ep in range(0, epochs):
                self.projector2.train()
                self.projector3.train()
                self.projector4.train()
                for i, (x, _, _, _) in enumerate(self.trainloader):
                    x = x.to(self.device)
                    out_a1, out_a2 = self.get_agent_out(x)

                    # project_out2 = self.projector2(out_a1["out2"].detach())
                    # loss2 = torch.mean((out_a2["out2"].detach() - project_out2) ** 2)
                    project_out21, project_out22 = self.projector2(out_a1["out2"].detach(), out_a2["out2"].detach())
                    loss21 = torch.mean((out_a1["out2"] - project_out21) ** 2)
                    loss22 = torch.mean((out_a2["out2"] - project_out22) ** 2)
                    loss2 = loss21 + loss22
                    self.optimizer2.zero_grad()
                    loss2.backward()
                    self.optimizer2.step()

                    project_out31, project_out32 = self.projector3(out_a1["out3"].detach(), out_a2["out3"].detach())
                    loss31 = torch.mean((out_a1["out3"].detach() - project_out31) ** 2)
                    loss32 = torch.mean((out_a2["out3"].detach() - project_out32) ** 2)
                    loss3 = loss31 + loss32
                    self.optimizer3.zero_grad()
                    loss3.backward()
                    self.optimizer3.step()

                    project_out41, project_out42 = self.projector4(out_a1["out4"].detach(), out_a2["out4"].detach())
                    loss41 = torch.mean((out_a1["out4"].detach() - project_out41) ** 2)
                    loss42 = torch.mean((out_a2["out4"].detach() - project_out42) ** 2)
                    loss4 = loss41 + loss42
                    self.optimizer4.zero_grad()
                    loss4.backward()
                    self.optimizer4.step() 

                    print(f"Epoch-{ep}-Step-{i}, {self.class_name} | loss2: {loss2.item():.6f} | loss3: {loss3.item():.6f} | loss4: {loss4.item():.6f}")
                    self.writer.add_scalar('Train/loss2', loss2.item(), ep*len(self.trainloader)+i)
                    self.writer.add_scalar('Train/loss3', loss3.item(), ep*len(self.trainloader)+i)
                    self.writer.add_scalar('Train/loss4', loss4.item(), ep*len(self.trainloader)+i)
                    # print(f"Epoch-{ep}-Step-{i}, {self.class_name} | loss: {loss.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
                    # self.writer.add_scalar('Train/loss_l2', loss1.item(), ep*len(self.trainloader)+i)
                    # self.writer.add_scalar('Train/loss_norm_l2', loss2.item(), ep*len(self.trainloader)+i)

                torch.save(self.projector2.state_dict(), self.ckpt2)
                torch.save(self.projector3.state_dict(), self.ckpt3)
                torch.save(self.projector4.state_dict(), self.ckpt4)
                if ep % 10 == 0:
                    metrix = self.test(cal_pro=False)
                    logger.info(f"Epoch-{ep}, {self.class_name} | all: {metrix['all'][0]:.5f}, {metrix['all'][1]:.5f} | 2: {metrix['2'][0]:.5f}, {metrix['2'][1]:.5f}"
                                f"| 3: {metrix['3'][0]:.5f}, {metrix['3'][1]:.5f} | 4: {metrix['4'][0]:.5f}, {metrix['4'][1]:.5f}")
                    self.writer.add_scalar('Val/imge_auc2', metrix['2'][0], ep)
                    self.writer.add_scalar('Val/pixel_auc2', metrix['2'][1], ep)
                    self.writer.add_scalar('Val/imge_auc3', metrix['3'][0], ep)
                    self.writer.add_scalar('Val/pixel_auc3', metrix['3'][1], ep)
                    self.writer.add_scalar('Val/imge_auc4', metrix['4'][0], ep)
                    self.writer.add_scalar('Val/pixel_auc4', metrix['4'][1], ep)
                    self.writer.add_scalar('Val/imge_auc', metrix['all'][0], ep)
                    self.writer.add_scalar('Val/pixel_auc', metrix['all'][1], ep)
            self.writer.close()
        else:
            pass

    def test(self, cal_pro=False):
        self.load_project_model()
        self.projector2.eval()
        self.projector3.eval()
        self.projector4.eval()
        with torch.no_grad():

            test_y_list = []
            test_mask_list = []
            test_img_list = []
            test_img_name_list = []
            # pixel-level
            score_map_list = []
            score_list = []

            score2_map_list = []
            score2_list = []
            score3_map_list = []
            score3_list = []
            score4_map_list = []
            score4_list = []

            for x, y, mask, name in self.testloader:
                test_y_list.extend(y.detach().cpu().numpy())
                test_mask_list.extend(mask.detach().cpu().numpy())
                test_img_list.extend(x.detach().cpu().numpy())
                test_img_name_list.extend(name)

                x = x.to(self.device)
                _, _, H, W = x.shape
                out_a1, out_a2 = self.get_agent_out(x)

                project_out21, project_out22 = self.projector2(out_a1["out2"], out_a2["out2"])
                loss21_map = torch.sum((out_a1["out2"] - project_out21) ** 2, dim=1, keepdim=True)
                loss22_map = torch.sum((out_a2["out2"] - project_out22) ** 2, dim=1, keepdim=True)
                loss2_map = (loss21_map + loss22_map) / 2.0
                score2_map = F.interpolate(loss2_map, size=(H, W), mode='bilinear', align_corners=False)
                score2_map = score2_map.cpu().detach().numpy()
                score2_map_list.extend(score2_map)
                score2_list.extend(np.squeeze(np.max(np.max(score2_map, axis=2), axis=2), 1))

                project_out31, project_out32 = self.projector3(out_a1["out3"], out_a2["out3"])
                loss31_map = torch.sum((out_a1["out3"] - project_out31) ** 2, dim=1, keepdim=True)
                loss32_map = torch.sum((out_a2["out3"] - project_out32) ** 2, dim=1, keepdim=True)
                loss3_map = (loss31_map + loss32_map) / 2.0
                score3_map = F.interpolate(loss3_map, size=(H, W), mode='bilinear', align_corners=False)
                score3_map = score3_map.cpu().detach().numpy()
                score3_map_list.extend(score3_map)
                score3_list.extend(np.squeeze(np.max(np.max(score3_map, axis=2), axis=2), 1))


                project_out41, project_out42 = self.projector4(out_a1["out4"], out_a2["out4"])
                loss41_map = torch.sum((out_a1["out4"] - project_out41) ** 2, dim=1, keepdim=True)
                loss42_map = torch.sum((out_a2["out4"] - project_out42) ** 2, dim=1, keepdim=True)
                loss4_map = (loss41_map + loss42_map) / 2.0
                score4_map = F.interpolate(loss4_map, size=(H, W), mode='bilinear', align_corners=False)
                score4_map = score4_map.cpu().detach().numpy()
                score4_map_list.extend(score4_map)
                score4_list.extend(np.squeeze(np.max(np.max(score4_map, axis=2), axis=2), 1))

                score_map = (score4_map + score3_map + score2_map) / 3
                score_map_list.extend(score_map)
                score_list.extend(np.squeeze(np.max(np.max(score_map, axis=2), axis=2), 1))

            visualize(test_img_list, test_mask_list, score_map_list, test_img_name_list, self.class_name,
                      f"{self.save_root}image/", 10000)
            # ROCAUC
            # imge_auc2, pixel_auc2, pixel_pro2 = self.cal_auc(score2_list, score2_map_list, test_y_list, test_mask_list)
            # imge_auc3, pixel_auc3, pixel_pro3 = self.cal_auc(score3_list, score3_map_list, test_y_list, test_mask_list)
            # imge_auc4, pixel_auc4, pixel_pro4 = self.cal_auc(score4_list, score4_map_list, test_y_list, test_mask_list)
            imge_auc, pixel_auc, pixel_pro = self.cal_auc(score_list, score_map_list, test_y_list, test_mask_list)
            # print(f"pixel AUC: {pixel_level_ROCAUC:.5f}")
            # metrix = {"2": [imge_auc2, pixel_auc2, pixel_pro2], "3": [imge_auc3, pixel_auc3, pixel_pro3],"4":[imge_auc4,
            #           pixel_auc4, pixel_pro4], "all": [imge_auc, pixel_auc, pixel_pro]}
            metrix = { "all": [imge_auc, pixel_auc, pixel_pro]}
            return metrix

    def cal_auc(self, score_list, score_map_list, test_y_list, test_mask_list):
        flatten_y_list = np.array(test_y_list).ravel()
        flatten_score_list = np.array(score_list).ravel()
        image_level_ROCAUC = roc_auc_score(flatten_y_list, flatten_score_list)

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list) 
        pro_auc_score = cal_pro_metric_new(test_mask_list, score_map_list, fpr_thresh=0.3)
        return image_level_ROCAUC, pixel_level_ROCAUC, pro_auc_score

    def load_project_model(self):
        self.projector2.load_state_dict(torch.load(self.ckpt2))
        self.projector3.load_state_dict(torch.load(self.ckpt3))
        self.projector4.load_state_dict(torch.load(self.ckpt4))



def parse_args():
    parser = argparse.ArgumentParser('STFPM_Center')
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--train", type=bool, default=True)

    parser.add_argument("--data_trans", type=str, default='imagenet', choices=['navie', 'imagenet'])

    parser.add_argument("--loss_type", type=str, default='l2norm+l2',
                        choices=['l2norm+l2', 'l2', 'l1', 'consine', 'l2+consine'])

    parser.add_argument("--model_name", type=str, default='vgg16', choices=['vgg16', 'resnet18'])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)  # 6 or 20 for train
    parser.add_argument("--lr2", type=float, default=3e-3)
    parser.add_argument("--lr3", type=float, default=3e-4)
    parser.add_argument("--lr4", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--latent_dim", type=int, default=200)

    parser.add_argument("--data_root", type=str, default="D:/Dataset/mvtec_anomaly_detection/")
    parser.add_argument("--resize", type=int, default=256)

    parser.add_argument("--post_smooth", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    set_seed(args.seed)
    logger.add(
        f'./result/MB-PFM-VGG_{args.seed}/logger-{args.data_trans}-{args.loss_type}-{args.resize}-{args.model_name}.txt',
        rotation="200 MB",
        backtrace=True,
        diagnose=True)
    logger.info(str(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    from dataset.mvtec import MVTecDataset, MVTec_CLASS_NAMES
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from PIL import Image

    if args.data_trans == 'navie':
        trans_x = T.Compose([T.Resize(args.resize, Image.ANTIALIAS),
                             T.ToTensor()])
    else:
        trans_x = T.Compose([T.Resize(args.resize, Image.ANTIALIAS),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    image_aucs = []
    pixel_aucs = []
    pro_30s = []
    for class_name in MVTec_CLASS_NAMES:
        torch.cuda.empty_cache()
        trainset = MVTecDataset(root_path=args.data_root, is_train=True, class_name=class_name, resize=args.resize,
                                trans=trans_x)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)

        testset = MVTecDataset(root_path=args.data_root, is_train=False, class_name=class_name, resize=args.resize,
                               trans=trans_x)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        model = DFP_AD(type="vgg")
        model.register(class_name=class_name, trainloader=trainloader, testloader=testloader, loss_type=args.loss_type,
                       data_trans=args.data_trans, size=args.resize, device=device,
                       latent_dim=args.latent_dim,
                       lr2=args.lr2, lr3=args.lr3, lr4=args.lr4, weight_decay=args.weight_decay,
                       seed=args.seed)
        if args.train:
            model.train(epochs=args.epochs)
        # else:
        # model.load_student_weight()
        metrix = model.test(cal_pro=True)
        image_aucs.append(metrix["all"][0])
        pixel_aucs.append(metrix["all"][1])
        pro_30s.append(metrix["all"][2])
        logger.info(f"{class_name}, image auc: {metrix['all'][0]:.5f}, pixel auc: {metrix['all'][1]:.5f}, pixel pro0.3: {metrix['all'][2]:.5f}")

    i_auc = np.mean(np.array(image_aucs))
    p_auc = np.mean(np.array(pixel_aucs))
    pro_auc = np.mean(np.array(pro_30s))
    logger.info(f"total, image AUC: {i_auc:.5f} | pixel AUC: {p_auc:.5f} | pixel PROo.3: {pro_auc:.5f}")
