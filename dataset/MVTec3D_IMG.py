import cv2.cv2
import matplotlib

matplotlib.use("Agg")
import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from torchvision import transforms
from imageio import imsave
# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

MVTEC_CLASSES=['bagel', 'cable_gland','carrot', 'cookie','dowel',
               'peach', 'potato', 'rope',  'tire' , 'foam']

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x


class MVTec3DDataset_IMG(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'test')

        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type,'rgb') + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type,'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, 'gt') + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        names = img_path.split("\\")
        name = names[-3] + "_" + names[-1]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, label, gt, name





if __name__ == '__main__':
    categories = MVTEC_CLASSES
    dataset_path = r'../datasets/mvtec_anomaly_detection'
    load_size = 256
    input_size = 256
    data_transforms = transforms.Compose([
        transforms.Resize((load_size, load_size), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((load_size, load_size), Image.NEAREST),
        transforms.ToTensor(),
        transforms.CenterCrop(input_size)])

    for category in categories:
        phase = 'train'
        dataset = MVTecDatasetSpxl(root=os.path.join(dataset_path, category),
                                transform=data_transforms, gt_transform=gt_transforms, phase=phase)

        for img, gt, label, name, img_type, spxl_label in dataset:
            save_folder = os.path.join(dataset_path, 'spxls', category, phase)
            save_name = os.path.join(save_folder, f'{name}.bmp')
            os.makedirs(save_folder, exist_ok=True)


            pass


    pass
