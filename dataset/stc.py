import os
import glob
import cv2
import numpy as np
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


DATA_ROOT = "D:/Dataset/ShanghaiTech/"
BIN_ROOT = os.path.join(DATA_ROOT, 'bin')
os.makedirs(BIN_ROOT, exist_ok=True)

INTERVAL = 5

def capture_training_frames():

    video_path = os.path.join(DATA_ROOT, 'training', 'videos')
    video_list = glob.glob(video_path + '/*.avi')
    video_list.sort()

    frame_root = os.path.join(BIN_ROOT, "train", "image")

    for v in video_list:
        video_name = os.path.basename(v)

        scene_class = video_name.split('_')[0]
        frame_path = os.path.join(frame_root, scene_class)
        os.makedirs(frame_path, exist_ok=True)

        cap = cv2.VideoCapture(v)
        cnt = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if cnt % INTERVAL == 0:
                # cv2.imshow(video_name, frame)
                # cv2.waitKey(0)
                frame = cv2.resize(frame, (256, 256))
                frame_name = video_name.split('.')[0] + f"_{cnt}.png"
                cv2.imwrite(os.path.join(frame_path, frame_name), frame)
                logger.info(os.path.join(frame_path, frame_name))
            cnt += 1
        cap.release()

def split_testing_frames():
    testing_frames_root = os.path.join(DATA_ROOT, "testing", "frames")
    testing_frame_mask_root = os.path.join(DATA_ROOT, "testing", "test_frame_mask")
    testing_pixel_mask_root = os.path.join(DATA_ROOT, "testing", "test_pixel_mask")

    img_root = os.path.join(BIN_ROOT, "test", 'image')
    gt_root = os.path.join(BIN_ROOT, "test", 'groundtruth')

    scene_folders = os.listdir(testing_frames_root)
    scene_folders.sort()
    for sf in scene_folders:
        scenne_class = sf.split("_")[0]
        img_path = os.path.join(img_root, scenne_class)
        os.makedirs(img_path, exist_ok=True)
        gt_path = os.path.join(gt_root, scenne_class)
        os.makedirs(gt_path, exist_ok=True)

        frames_path = os.path.join(testing_frames_root, sf)
        frames_list = glob.glob(frames_path + "/*.*")
        frames_list.sort()

        frames_pixel_masks = np.load(os.path.join(testing_pixel_mask_root, sf + ".npy"))
        # print(np.max(frames_pixel_masks))
        for cnt, f in enumerate(frames_list):
            if cnt % 1 == 0:
                # frame
                frame = cv2.imread(f)
                frame = cv2.resize(frame, (256, 256))
                frame_name = os.path.basename(f).split('.')[0] + '.png'
                cv2.imwrite(os.path.join(img_path, frame_name), frame)
                logger.info(os.path.join(img_path, frame_name))

                # gt
                gt = frames_pixel_masks[cnt] * 255
                gt = cv2.resize(gt, (256, 256), cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(gt_path, frame_name), gt)


STC_CLASS = ["02", "01", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
# STC_CLASS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
class STCDataset(Dataset):
    def __init__(self, root_path=BIN_ROOT, scene_class='01', is_train=True,
                 resize=256, trans=None):

        assert scene_class in STC_CLASS

        self.root_path = root_path
        self.scene_class = scene_class
        self.is_train = is_train

        # set transforms
        if trans is None:
            self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
        else:
            self.transform_x = trans
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.ToTensor()])

        self.x, self.mask = self.load_image()
        # self.len = len(self.x) // 3
        self.len = len(self.x)

    def load_image(self):
        if self.is_train:
            img_path = os.path.join(self.root_path, "train", "image", self.scene_class)
            img_list = glob.glob(img_path + "/*.png")
            mask_list = None
        else:
            img_path = os.path.join(self.root_path, "test", "image", self.scene_class)
            mask_path = os.path.join(self.root_path, "test", "groundtruth", self.scene_class)
            img_list = glob.glob(img_path + "/*.png")
            mask_list = glob.glob(mask_path + "/*.png")
            img_list.sort()
            mask_list.sort()
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
            _, H, W = x.shape
            mask = torch.zeros((1, H, W))
        else:
            mask = self.mask[idx]
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        y = torch.max(mask)
        return x, y, mask, name

if __name__ == "__main__":
    # capture_training_frames()
    # split_testing_frames()
    stc = STCDataset(is_train=False)
    data = stc[120]
    print(data)