# Training
# python MB-PFM-ResNet.py --train --gpu_id 0 --batch_size 8 --epochs 200 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/

# Testing
python MB-PFM-ResNet.py --gpu_id 0 --batch_size 1 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/