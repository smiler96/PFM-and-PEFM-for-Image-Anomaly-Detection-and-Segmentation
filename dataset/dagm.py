import os
import glob
import cv2
import numpy as np
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


DATA_ROOT = "D:/Dataset/DAGM2007/"
DAGMM_CLASS = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10"]
# DAGMM_CLASS = [ "Class1", "Class2", "Class3", "Class4", "Class5", "Class6"]
class DAGMDataset(Dataset):
    def __init__(self, root_path=DATA_ROOT, class_name='Class1', is_train=True,
                 resize=256):

        assert class_name in DAGMM_CLASS

        self.resize = resize
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.ToTensor()])

        self.x, self.mask = self.load_image()
        self.len = len(self.x)

    def load_image(self):
        phase = "Train" if self.is_train else "Test"
        img_path = os.path.join(self.root_path, self.class_name, phase)
        label_path = os.path.join(self.root_path, self.class_name, phase, "Label")
        label_file = os.path.join(label_path, "Labels.txt")
        with open(label_file, 'r') as f:
            info = f.readlines()
        info = info[1:]
        for i in range(len(info)):
            info[i] = info[i].strip('\n').split('\t')
        # print(info)

        mask_list = []
        img_list = []
        if self.is_train:
            for s in info:
                if s[1] == '0':
                    img_list.append(os.path.join(img_path, s[2]))
        else:
            for s in info:
                img_list.append(os.path.join(img_path, s[2]))
                if s[1] == "1":
                    mask_list.append(os.path.join(label_path, s[4]))
                else:
                    mask_list.append("None")
            # img_list.sort()
            # mask_list.sort()
            # self.len = len(img_list) // 3
            # new_img_list = img_list[len(img_list)-self.len:len(img_list)]
            # new_mask_list = mask_list[len(img_list)-self.len:len(img_list)]
            #
            # new_mask_list[self.len//2:self.len] = mask_list[len(img_list)-self.len//2:len(img_list)]
            # new_img_list[self.len//2:self.len] = img_list[len(img_list)-self.len//2:len(img_list)]
            #
            # new_mask_list[:self.len//2] = mask_list[:self.len//2]
            # new_img_list[:self.len//2] = img_list[:self.len//2]
            #
            # img_list = new_mask_list
            # mask_list = new_mask_list
            assert len(img_list) == len(mask_list)
        return img_list, mask_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        name = os.path.basename(x)
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if self.is_train:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = self.mask[idx]
            if mask != "None":
                mask = Image.open(mask)
                mask = self.transform_mask(mask)
                y = 1
            else:
                mask = torch.zeros([1, x.shape[1], x.shape[2]])
                y = 0

        return x, y, mask, name

if __name__ == "__main__":
    stc = DAGMDataset(is_train=False)
    data = stc[0]
    print(data)