# PFM and PEFM for Image Anomaly Detection and Segmentation

## Abstract

### Unsupervised Image Anomaly Detection and Segmentation Based on Pre-trained Feature Mapping [(PFM-TII)](https://ieeexplore.ieee.org/document/xxx)
Image anomaly detection and segmentation are important for the development of automatic product quality inspection in intelligent manufacturing. Because the normal data can be collected easily and abnormal ones are rarely existent, unsupervised methods based on reconstruction and embedding have been mainly studied for anomaly detection. But the detection performance and computing time requires to be further improved. This paper proposes a novel framework, named as Pre-trained Feature Mapping (PFM), for unsupervised image anomaly detection and segmentation. The proposed PFM maps the image from a pre-trained feature space to another one to detect the anomalies effectively. The bidirectional and multi-hierarchical bidirectional pre-trained feature mapping are further proposed and studied for improving the performance. The proposed framework achieves the better results on well-known MVTec AD dataset compared with state-of-the-art methods, with the area under the receiver operating characteristic curve of 97.5% for anomaly detection and of 97.3% for anomaly segmentation over all 15 categories. The proposed framework is also superior in term of the computing time. The extensive experiments on ablation studies are also conducted to show the effectiveness and efficient of the proposed framework.

### Position Encoding Enhanced Feature Mapping for Image Anomaly Detection [(PEFM CASE)](https://ieeexplore.ieee.org/document/xxx)
Image anomaly detection is an important stage for automatic visual inspection in intelligent manufacturing systems. The wide-ranging anomalies in images, such as various sizes, shapes, and colors, make automatic visual inspection challenging. Previous work on image anomaly detection has achieved significant advancements. However,  there are still limitations in terms of detection performance and efficiency. In this paper, a novel Position Encoding enhanced Feature Mapping (PEFM) method is proposed to address the problem of image anomaly detection, detecting the anomalies by mapping a pair of pre-trained features embedded with position encodes. Experiment results show that the proposed PEFM achieves better performance and efficiency than the state-of-the-art methods on the MVTec AD dataset, an AUCROC of 98.30% and an AUCPRO of 95.52%, and achieves the  AUCPRO of 94.0% on the MVTec 3D AD dataset. 

## Using

### PFM (TII)
```python
# Training
python MB-PFM-ResNet.py --train --gpu_id 0 --batch_size 8 --epochs 200 --lr 3e-4 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
# Testing
python MB-PFM-ResNet.py --gpu_id 0 --batch_size 1 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
```


### PEFM (CASE)

```python
# Training
python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/

# Testing
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
```

## Citation

If there is any help for your work, please consider citing this paper:

```
@ARTICLE{PFM,
    author={Wan, Qian and Gao, Liang and Li, Xinyu and Wen, Long},
    journal={IEEE Transactions on Industrial Informatics},
    title={Unsupervised Image Anomaly Detection and Segmentation Based on Pre-trained Feature Mapping},
    year={2022},
    volume={},
    number={},
    pages={},
    doi={10.1109/TII.2022.3182385}
} 
@INPROCEEDINGS{PEFM,
    author={Wan, Qian and Cao YunKang and Gao, Liang and Shen Weiming and Li, Xinyu},
    booktitle={2022 IEEE 18th International Conference on Automation Science and Engineering (CASE)}, 
    title={Position Encoding Enhanced Feature Mapping for Image Anomaly Detection}, 
    year={2022},
    volume={},
    number={},
    pages={},
    doi={}
  }
```

## Acknowledgment

Thanks for the excellent work for [SPADE](https://github.com/byungjae89/SPADE-pytorch).

