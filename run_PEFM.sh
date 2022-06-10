# Training
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --lr 3e-4 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_anomaly_detection/

# Testing
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --data_root D:/Dataset/mvtec_anomaly_detection/