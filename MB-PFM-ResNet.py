'''
    Unsupervised Image Anomaly Detection and Segmentation Based on Pre-trained Feature Mapping
'''
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.vgg import vgg16_bn, vgg19_bn
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import visualize, set_seed, cal_pro_metric_new
from loguru import logger
import argparse
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



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
    def __init__(self, agent_S='resnet50', agent_T="resnet101"):
        self.s_name = agent_S
        self.t_name = agent_T
        if agent_S == "resnet18" or agent_S == "resnet34":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [64, 128, 256]
            # self.outdim = [50, 100, 200]
        elif agent_S == "resnet50":
            self.Agent1 = PretrainedModel(model_name=agent_S)
            self.indim = [256, 512, 1024]
            # self.Agent2 = PretrainedModel(model_name="resnet34")
        if agent_T == "resnet50" or agent_T == "resnet101":
            # self.Agent1 = PretrainedModel(model_name="vgg16")
            self.Agent2 = PretrainedModel(model_name=agent_T)
            self.outdim = [256, 512, 1024]
            self.latent_dim = [200, 400, 900]

    def register(self, **kwargs):
        self.class_name = kwargs['class_name']
        self.device = kwargs['device']
        self.trainloader = kwargs['trainloader']
        self.testloader = kwargs['testloader']

        self.projector2 = DualProjectionNet(in_dim=self.indim[0], out_dim=self.outdim[0], latent_dim=self.latent_dim[0])
        self.optimizer2 = torch.optim.Adam(self.projector2.parameters(), lr=kwargs["lr2"], weight_decay=kwargs["weight_decay"])
        self.projector3 = DualProjectionNet(in_dim=self.indim[1], out_dim=self.outdim[1], latent_dim=self.latent_dim[1])
        self.optimizer3 = torch.optim.Adam(self.projector3.parameters(), lr=kwargs["lr3"], weight_decay=kwargs["weight_decay"])
        self.projector4 = DualProjectionNet(in_dim=self.indim[2], out_dim=self.outdim[2], latent_dim=self.latent_dim[2])
        self.optimizer4 = torch.optim.Adam(self.projector4.parameters(), lr=kwargs["lr4"], weight_decay=kwargs["weight_decay"])

        self.Agent1.to(self.device).eval()
        self.Agent2.to(self.device).eval()

        self.projector2.to(self.device)
        self.projector3.to(self.device)
        self.projector4.to(self.device)

        self.save_root = "./result/MB-PFM_{}-{}_{}/".format(self.s_name, self.t_name, kwargs["seed"])
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


    def statistic_var(self, c=False):
        if c:
            self.var21 = 0
            self.var22 = 0
            self.var31 = 0
            self.var32 = 0
            self.var41 = 0
            self.var42 = 0
            with torch.no_grad():
                for i, (x, _, _, _) in enumerate(self.trainloader):
                    torch.cuda.empty_cache()
                    x = x.to(self.device)
                    out_a1, out_a2 = self.get_agent_out(x)
                    project_out21, project_out22 = self.projector2(out_a1["out2"], out_a2["out2"])
                    var21 = (out_a1["out2"] - project_out21) ** 2
                    var22 = (out_a2["out2"] - project_out22) ** 2
                    self.var21 += torch.mean(var21, dim=0, keepdim=True)
                    self.var22 += torch.mean(var22, dim=0, keepdim=True)

                    project_out31, project_out32 = self.projector3(out_a1["out3"], out_a2["out3"])
                    var31 = (out_a1["out3"] - project_out31) ** 2
                    var32 = (out_a2["out3"] - project_out32) ** 2
                    self.var31 += torch.mean(var31, dim=0, keepdim=True)
                    self.var32 += torch.mean(var32, dim=0, keepdim=True)

                    project_out41, project_out42 = self.projector4(out_a1["out4"], out_a2["out4"])
                    var41 = (out_a1["out4"] - project_out41) ** 2
                    var42 = (out_a2["out4"] - project_out42) ** 2
                    self.var41 += torch.mean(var41, dim=0, keepdim=True)
                    self.var42 += torch.mean(var42, dim=0, keepdim=True)

            self.var21 /= len(self.trainloader)
            self.var22 /= len(self.trainloader)
            self.var31 /= len(self.trainloader)
            self.var32 /= len(self.trainloader)
            self.var41 /= len(self.trainloader)
        else:
            self.var21 = 1
            self.var22 = 1
            self.var31 = 1
            self.var32 = 1
            self.var41 = 1
            self.var42 = 1


    def test(self, cal_pro=False):
        self.load_project_model()
        self.projector2.eval()
        self.projector3.eval()
        self.projector4.eval()

        self.statistic_var()

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

            start_t = time.time()
            for x, y, mask, name in self.testloader:
                test_y_list.extend(y.detach().cpu().numpy())
                test_mask_list.extend(mask.detach().cpu().numpy())
                test_img_list.extend(x.detach().cpu().numpy())
                test_img_name_list.extend(name)

                x = x.to(self.device)
                _, _, H, W = x.shape
                out_a1, out_a2 = self.get_agent_out(x)

                project_out21, project_out22 = self.projector2(out_a1["out2"], out_a2["out2"])
                loss21_map = torch.sum((out_a1["out2"] - project_out21) ** 2 / self.var21, dim=1, keepdim=True)
                loss22_map = torch.sum((out_a2["out2"] - project_out22) ** 2 / self.var22, dim=1, keepdim=True)

                loss2_map = (loss21_map + loss22_map) / 2.0
                score2_map = F.interpolate(loss2_map, size=(H, W), mode='bilinear', align_corners=False)
                score2_map = score2_map.cpu().detach().numpy()
                score2_map_list.extend(score2_map)
                score2_list.extend(np.squeeze(np.max(np.max(score2_map, axis=2), axis=2), 1))

                project_out31, project_out32 = self.projector3(out_a1["out3"], out_a2["out3"])
                loss31_map = torch.sum((out_a1["out3"] - project_out31) ** 2 / self.var31, dim=1, keepdim=True)
                loss32_map = torch.sum((out_a2["out3"] - project_out32) ** 2 / self.var32, dim=1, keepdim=True)

                loss3_map = (loss31_map + loss32_map) / 2.0
                score3_map = F.interpolate(loss3_map, size=(H, W), mode='bilinear', align_corners=False)
                score3_map = score3_map.cpu().detach().numpy()
                score3_map_list.extend(score3_map)
                score3_list.extend(np.squeeze(np.max(np.max(score3_map, axis=2), axis=2), 1))


                project_out41, project_out42 = self.projector4(out_a1["out4"], out_a2["out4"])
                loss41_map = torch.sum((out_a1["out4"] - project_out41) ** 2 / self.var41, dim=1, keepdim=True)
                loss42_map = torch.sum((out_a2["out4"] - project_out42) ** 2 / self.var42, dim=1, keepdim=True)

                loss4_map = (loss41_map + loss42_map) / 2.0
                score4_map = F.interpolate(loss4_map, size=(H, W), mode='bilinear', align_corners=False)
                score4_map = score4_map.cpu().detach().numpy()
                score4_map_list.extend(score4_map)
                score4_list.extend(np.squeeze(np.max(np.max(score4_map, axis=2), axis=2), 1))

                score_map = (score4_map + score3_map + score2_map) / 3
                # score_map = gaussian_filter(score_map.squeeze(), sigma=4)

                score_map_list.extend(score_map)
                # score_list.extend(np.squeeze(np.max(np.max(score_map, axis=2), axis=2), 1))
                score_map = np.reshape(score_map, (score_map.shape[0], -1))
                score_list.extend(np.max(score_map, 1))

            end_t = time.time()
            t_per_imge = end_t - start_t
            t_per_imge = t_per_imge / len(score_list)

            visualize(test_img_list, test_mask_list, score_map_list, test_img_name_list, self.class_name,
                      f"{self.save_root}image/", 10000)
 
            # ROCAUC
            # imge_auc2, pixel_auc2, pixel_pro2 = self.cal_auc(score2_list, score2_map_list, test_y_list, test_mask_list)
            # imge_auc3, pixel_auc3, pixel_pro3 = self.cal_auc(score3_list, score3_map_list, test_y_list, test_mask_list)
            # imge_auc4, pixel_auc4, pixel_pro4 = self.cal_auc(score4_list, score4_map_list, test_y_list, test_mask_list)
            imge_auc, pixel_auc, pixel_pro = self.cal_auc(score_list, score_map_list, test_y_list, test_mask_list)
            # print(f"pixel AUC: {pixel_level_ROCAUC:.5f}")
            # metrix = {"2": [imge_auc2, pixel_auc2, pixel_pro2], "3": [imge_auc3, pixel_auc3, pixel_pro3],"4":[imge_auc4,
            #           pixel_auc4, pixel_pro4], "all": [imge_auc, pixel_auc, pixel_pro], "time": t_per_imge}
            metrix = {"all": [imge_auc, pixel_auc, pixel_pro], "time": t_per_imge}

 
            # test_y_list = np.array(test_y_list)
            # score_list = np.array(score_list)
            # p_index = test_y_list == 1
            # n_index = test_y_list == 0

            # p_score = score_list[p_index]
            # n_score = score_list[n_index]
            # data = [n_score, p_score]
            # np.save(os.path.join(self.save_root, f"{self.class_name}_image.npy"), (p_score, n_score))
            # # import seaborn as sns
            # # sns.boxplot(data=data)
            # # sns.histplot(p_score, kde=False, color="r")
            # # sns.histplot(n_score, kde=False, color="b")

            # # plt.show()
            return metrix

    def cal_auc(self, score_list, score_map_list, test_y_list, test_mask_list):
        flatten_y_list = np.array(test_y_list).ravel()
        flatten_score_list = np.array(score_list).ravel()
        image_level_ROCAUC = roc_auc_score(flatten_y_list, flatten_score_list)

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list)
        # pro_auc_score = 0 
        pro_auc_score = cal_pro_metric_new(test_mask_list, score_map_list, fpr_thresh=0.3)
        return image_level_ROCAUC, pixel_level_ROCAUC, pro_auc_score

    def load_project_model(self):
        self.projector2.load_state_dict(torch.load(self.ckpt2))
        self.projector3.load_state_dict(torch.load(self.ckpt3))
        self.projector4.load_state_dict(torch.load(self.ckpt4))

def center_crop(img, dim):

    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def parse_args():
    parser = argparse.ArgumentParser('STFPM_Center')
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--train", action="store_false")

    parser.add_argument("--data_trans", type=str, default='imagenet', choices=['navie', 'imagenet'])

    parser.add_argument("--loss_type", type=str, default='l2norm+l2',
                        choices=['l2norm+l2', 'l2', 'l1', 'consine', 'l2+consine'])

    parser.add_argument("--agent_S", type=str, default='resnet34')
    parser.add_argument("--agent_T", type=str, default='resnet50')

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)  
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
        f'./result/MB-PFM_{args.agent_S}-{args.agent_T}_{args.seed}/logger-{args.resize}.txt',
        rotation="200 MB",
        backtrace=True,
        diagnose=True)
    logger.info(str(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

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
    times = []
    # plt.figure(figsize=(10, 8))
    for class_name in MVTec_CLASS_NAMES:
        torch.cuda.empty_cache()
        trainset = MVTecDataset(root_path=args.data_root, is_train=True, class_name=class_name, resize=args.resize,
                                trans=trans_x)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)

        testset = MVTecDataset(root_path=args.data_root, is_train=False, class_name=class_name, resize=args.resize,
                               trans=trans_x)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        model = DFP_AD(agent_S=args.agent_S, agent_T=args.agent_T)
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
        logger.info(f"{class_name}, image auc: {metrix['all'][0]:.4f}, pixel auc: {metrix['all'][1]:.4f}, pixel pro0.3: {metrix['all'][2]:.4f}")
        times.append(metrix['time'])
        logger.info(f"{class_name}, time: {metrix['time']:.5f}s.")

    i_auc = np.mean(np.array(image_aucs))
    p_auc = np.mean(np.array(pixel_aucs))
    pro_auc = np.mean(np.array(pro_30s))
    times = np.mean(np.array(times))
    logger.info(f"total, image AUC: {i_auc:.4f} | pixel AUC: {p_auc:.4f} | pixel PROo.3: {pro_auc:.4f}")
    logger.info(f"total, time: {times:.5f}s.")
