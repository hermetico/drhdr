#python train.py -bs 12 --model drhdr --gpu 0 --limit_training_dataset .5 --crop_train_data 256 --name demo-challenge  --loss mu --save_samples_every 25 --save_every 25 --epochs 150 --lr 0.0005 --dataset_dir "/mnt/hdd/shared/datasets/ntire-hdr-2022-clean-256/"
python train.py -bs 12 --model drhdr --gpu 0 --limit_training_dataset .05 --crop_train_data 256 --name demo-challenge  --loss mu --save_samples_every 1 --save_every 1 --epochs 1 --lr 0.0005 --dataset_dir "/mnt/hdd/shared/datasets/ntire-hdr-2022-clean-256/"