#!/bin/bash
# Usage:
# Run this script at reprlearn/scripts folder
# bash run_train_gan_mnist.sh

# Supported datamodules:
# maptiles, mnist, mnist_m,

# Support plmodules:
#
nohup python train_gm.py --model_name conv_fc_gan --data_name mnist \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnist \
--gpu_id 1 --max_epochs 500 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--num_generated_sample 10000 \
--save_sample \
--save_sample_snapshot \
--log_sample_to_tb \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M:%s)".txt &

# larger batch_size
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 64 --latent_emb_dim 128 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnist \
--gpu_id 1 --max_epochs 500 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# stronger generator (resnet)
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 64 --latent_emb_dim 128 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnist \
--gpu_id 1 --max_epochs 500 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &


# Train on MNIST-M (MNIST digits on random patch of background)
#todo: increase max-epochs
nohup python train_gm.py --model_name conv_fc_gan --data_name mnistm \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnistm --data_root '/data/hayley-old/Tenanbaum2000/data/MNIST-M' \
--gpu_id 1 --max_epochs 10 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--num_generated_sample 10000 \
--save_sample \
--save_sample_snapshot \
--log_sample_to_tb \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M:%s)".txt &

# Train on Mono_MNIST (MNIST digits on monochrome background)
#todo: increase max-epochs
nohup python train_gm.py --model_name conv_fc_gan --data_name mnistm \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mono_mnist --color red --data_root '/data/hayley-old/Tenanbaum2000/data/Mono-MNIST' \
--gpu_id 1 --max_epochs 10 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--num_generated_sample 10000 \
--save_sample \
--save_sample_snapshot \
--log_sample_to_tb \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M:%s)".txt &

# Train on USPS
#todo: increase max-epochs
nohup python train_gm.py --model_name conv_fc_gan --data_name mnistm \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name usps --data_root '/data/hayley-old/Tenanbaum2000/data/USPS' \
--gpu_id 1 --max_epochs 10 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--num_generated_sample 10000 \
--save_sample \
--save_sample_snapshot \
--log_sample_to_tb \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M:%s)".txt &

# Train  on usps (n_datapts: 9298; default size: (1,32,32))
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name usps \
--gpu_id 0 --max_epochs 100 --batch_size 64 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# todo: use larger batch_size
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name usps \
--gpu_id 0 --max_epochs 100 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 32 --latent_emb_dim 128 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name usps \
--gpu_id 0 --max_epochs 300 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# todo: experiment with resnet generator
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 32 --latent_emb_dim 128 \
--dec_type resnet --dec_hidden_dims 256 128 64 32 \
--data_name usps \
--gpu_id 2 --max_epochs 300 --batch_size 1028 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# todo: Train  on Multi Monochrome MNIST
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name mnistm \
--gpu_id 0 --max_epochs 3 --batch_size 32 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &


# todo: Train  on Maptile dataset
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 10 --latent_emb_dim 32 \
--dec_type conv --dec_hidden_dims 256 128 64 32 \
--data_name maptiles \
--cities la paris \
--styles CartoVoyagerNoLabels StamenTonerBackground \
--zooms 14 \
--gpu_id 0 --max_epochs 100 --batch_size 32 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# 11-30-2021
# train on EsriImagery maptiles, all cities
# on cropped 64x64 map tiles in /data/hayley-old/Maptiles-64x64-thresh:0.01
# --latent_dim = 1028
#--dec_hidden_dims 256 128 128 64 32
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 1028 --latent_emb_dim 4112 \
--dec_type conv --dec_hidden_dims 256 128 128 64 32 \
--data_name maptiles \
--data_root  /data/hayley-old/Maptiles-64x64-thresh:0.01 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
     'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
     'berlin' 'montreal' 'rome' \
--styles EsriImagery \
--zooms 14 \
--gpu_id 1 --max_epochs 1000 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

# even larger latent dim
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 2056 --latent_emb_dim 4112 \
--dec_type conv --dec_hidden_dims 256 128 128 64 32 \
--data_name maptiles \
--data_root  /data/hayley-old/Maptiles-64x64-thresh:0.01 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
     'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
     'berlin' 'montreal' 'rome' \
--styles EsriImagery \
--zooms 14 \
--gpu_id 1 --max_epochs 1000 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &



#resnet decoder
nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 1028 --latent_emb_dim 4112 \
--dec_type resnet --dec_hidden_dims 256 128 128 64 32 \
--data_name maptiles \
--data_root  /data/hayley-old/Maptiles-64x64-thresh:0.01 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
     'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
     'berlin' 'montreal' 'rome' \
--styles EsriImagery \
--zooms 14 \
--gpu_id 1 --max_epochs 1000 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &

nohup python train_gm.py --model_name conv_fc_gan  \
--latent_dim 2056 --latent_emb_dim 4112 \
--dec_type resnet --dec_hidden_dims 256 128 128 64 32 \
--data_name maptiles \
--data_root  /data/hayley-old/Maptiles-64x64-thresh:0.01 \
--cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
     'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
     'berlin' 'montreal' 'rome' \
--styles EsriImagery \
--zooms 14 \
--gpu_id 1 --max_epochs 1000 --batch_size 128 \
--lr_g 1e-4 --lr_d 1e-3 -k 1 \
--b1 0.5 --b2 0.9 \
--ckpt_metric "val/loss_G" \
--save_top_k 5 \
--no_early_stop \
--n_show 64 \
--log_root /data/hayley-old/Tenanbaum2000/lightning_logs/"$(date +%F)" \
2>&1 | tee log-"$(date +%F-%H:%M)".txt &












