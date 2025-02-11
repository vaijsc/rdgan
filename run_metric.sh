#!/bin/sh

# --------------- CIFAR ------------------
# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/cifar10 --mirror=1 \
# 	--gen_data=./wddgan_generated_samples/cifar10 --img_resolution=32

# --------------- Celeb ------------------
# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/celeba_256 --mirror=1 \
# 	--gen_data=./ddgan_generated_samples/celeba_256 --img_resolution=256

# wddgan_generated_samples/celeba_256
# generated_wddgan_ablation_full_quan_bs64/celeba_256/
# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/celeba_256 --mirror=1 \
# 	--gen_data=wddgan_generated_samples/celeba_256 --img_resolution=256


# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/celeba_512 --mirror=1 \
# 	--gen_data=generated_wddgan_ablation_full_quan_bs64/celeba_256/ --img_resolution=512

# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/celeba_512 --mirror=1 \
# 	--gen_data=ddgan_generated_samples/celeba_512 --img_resolution=512
# 

# --------------- LSUN ------------------
python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/lsun/ --mirror=1 \
	--gen_data=wddgan_generated_samples/lsun --img_resolution=256

# --------------- STL ------------------
# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/stl10/ --mirror=1 \
# 	--gen_data=ddgan_generated_samples/stl10 --img_resolution=64

# python calc_metrics.py --metrics=pr50k3_full --data=./real_samples/stl10/ --mirror=1 \
# 	--gen_data=wddgan_generated_samples/stl10 --img_resolution=64
