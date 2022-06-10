import os

from skimage import measure
from sklearn.metrics import auc
from loguru import logger

import cv2
import numpy as np

def save_score_map(score_maps, names, class_name, save_path, loss_name=None):
    assert loss_name is not  None
    num = len(score_maps)
    os.makedirs(save_path, exist_ok=True)
    score_maps = np.array(score_maps)
    max_mum = np.max(score_maps)
    min_mum = np.min(score_maps)
    score_maps = (score_maps - min_mum) / (max_mum - min_mum) * 255.0
    for _idx in range(num):
        score_map = np.uint8(np.squeeze(score_maps[_idx]))
        score_map = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)

        _name = names[_idx].split('.')[0]
        path0 = os.path.join(save_path, f"{class_name}_{_name}_{loss_name}.png")
        cv2.imwrite(path0, score_map)

def visualize(test_imgs, test_masks, score_maps, names,  class_name, save_path, num=100, trans='imagenet'):
    num = min(num, len(test_imgs))
    os.makedirs(save_path, exist_ok=True)
    score_maps = np.array(score_maps)
    max_mum = np.max(score_maps)
    min_mum = np.min(score_maps)
    score_maps = (score_maps - min_mum) / (max_mum - min_mum) * 255.0
    for _idx in range(num):
        test_img = test_imgs[_idx]
        test_img = denormalize(test_img, trans=trans)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

        test_mask = test_masks[_idx].transpose(1, 2, 0).squeeze()

        score_map = np.uint8(np.squeeze(score_maps[_idx]))
        # score_map = cv2.normalize(score_map, score_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        score_map = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)
        # cv2.imshow("score_map", score_map)
        # cv2.waitKey(0)
        # res_img = cv2.addWeighted(test_img, 0.4, score_map, 0.6, 0)
        # test_img = draw_detect(test_img, test_mask)

        name = names[_idx].split('.')[0]
        path0 = os.path.join(save_path, f"{class_name}_{name}.png")
        cv2.imwrite(path0, score_map)

        test_img_mask = draw_detect(test_img, test_mask)
        path1 = os.path.join(save_path, f"{class_name}_{name}_gt.png")
        cv2.imwrite(path1, test_img_mask)


def draw_detect(img, label):
    assert len(label.shape) == 2
    label = np.uint8(label*255)
    _, label = cv2.threshold(label, 5, 255, 0)
    _, label_cnts, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # img = cv2.drawContours(img, label_cnts, -1, (0, 0, 255), 1)
    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, label_cnts, color=(0, 0, 255))
    label=np.expand_dims(255-label, axis=2)/255
    img = img*(np.concatenate([label, label, label], axis=2)) + mask
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img


def denormalize(img, trans='imagenet'):
    if trans == 'imagenet':
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    elif trans == 'coco':
        std = np.array([0.5, 0.5, 0.5])
        mean = np.array([0.5, 0.5, 0.5])
        x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    elif trans == 'navie':
        x = (img.transpose(1, 2, 0) * 255.).astype(np.uint8)
    elif trans == 'no':
        x = (img.transpose(1, 2, 0)).astype(np.uint8)
    else:
        raise NotImplementedError
    return x

import os
from PIL import Image

from torchvision import transforms as T
def load_image(path):
    transform_x = T.Compose([T.Resize(256, Image.ANTIALIAS),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    x = Image.open(path).convert('RGB')
    x = transform_x(x)
    x = x.unsqueeze(0)
    return x
 

def cal_pro_metric_new(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
    labeled_imgs = np.array(labeled_imgs).squeeze(1)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)
    score_imgs = np.array(score_imgs).squeeze(1)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)


    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

 

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

import torch
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





from thop import profile
from thop import clever_format
def calculate_flops(device, xs, xt, model): 
    model = model.eval().to(device)
    xs = xs.to(device)
    xt = xt.to(device) 

    flops, params = profile(model, inputs=(xs, xt,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"[INFO] flops: {flops}")
    print(f"[INFO] params: {params}") 
    return flops, params


if __name__ == "__main__":
    from MDFP_Dual_Norm_AD import DualProjectionNet
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ins = [64, 128, 256]
    outs = [256, 512, 1024]
    latents = [200, 400, 900]
    xs = [torch.rand([1, 64, 64, 64]), torch.rand([1, 128, 32, 32]), torch.rand([1, 256, 16, 16])]
    xt = [torch.rand([1, 256, 64, 64]), torch.rand([1, 512, 32, 32]), torch.rand([1, 1024, 16, 16])]
    for _in, _out, _latent, _xs, _xt in zip(ins, outs, latents, xs, xt): 
        model1 = DualProjectionNet(in_dim=_in, out_dim=_out, latent_dim=_latent)
        flops, params = calculate_flops(device, model=model1, xs=_xs, xt=_xt)

