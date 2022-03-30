# Runs inference and metrics on the pretrained model
python inference.py --model drhdr --gpu 0 --name candidate --inference_test_dataset_dir "/mnt/hdd/shared/datasets/ntire-hdr-2022/ntire-test-input"
#python _dump_model.py --model drhdr --gpu 0 --name candidate --inference_test_dataset_dir "/mnt/hdd/shared/datasets/ntire-hdr-2022/ntire-test-input"
#python inference.py --model drhdr --gpu 0 --name challenge-demo --inference_test_dataset_dir "/mnt/hdd/shared/datasets/ntire-hdr-2022/ntire-test-input"
