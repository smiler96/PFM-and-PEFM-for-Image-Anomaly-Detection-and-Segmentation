# Training for MVTec AD
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root /home/dlwanqian/data/mvtec_anomaly_detection/
# python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root /home/dlwanqian/data/mvtec_anomaly_detection/

# Testing for MVTec AD
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 128 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root D:/Dataset/mvtec_anomaly_detection/
python PEFM_AD.py --gpu_id 0 --batch_size 1 --resize 512 --data_trans imagenet --loss_type l2norm+l2 --pe_required --data_root D:/Dataset/mvtec_anomaly_detection/


# Training for MVTec 3D AD
python PEFM_AD.py --train --gpu_id 0 --batch_size 16 --epochs 200 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_3d_anomaly_detection/

# Testing for MVTec 3D AD
python PEFM_AD.py --gpu_id 0 --batch_size 16 --epochs 200 --resize 256 --data_trans imagenet --loss_type l2norm+l2 --data_root /home/dlwanqian/data/mvtec_3d_anomaly_detection/