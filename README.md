# Training scripts
## Dataset-independence
- `train.py`: train one model (eg. beta-vae, IWAE, bivae) on one specific hyperparamter config
    - E.g. Train `BiVAE` on `osmnx_roads` data of the following `cities`, with images of `bgcolors` 
    ```bash
    nohup python train.py --model_name="bivae" \
    --latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 1.0 \
    --data_root="/data/hayley-old/osmnx_data/images" \
    --data_name="osmnx_roads" \
    --cities 'la' 'charlotte' 'vegas' 'boston' 'paris' \
         'amsterdam' 'shanghai' 'seoul' 'chicago' 'manhattan' \
         'berlin' 'montreal' 'rome' \
    --bgcolors "k" "r" "g" "b" "y" --n_styles=5 \
    --zooms 14 \
    --gpu_id=2 --max_epochs=300   --terminate_on_nan=True  \
    -lr 3e-4 -bs 32 \
    --log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-05-18/" &
    ```
  
    - E.g.: Train `BIVAE` on Rotated MNIST of optionally specified subset (given as a filepath to `.npy` file containing 
    the indices from the original Training MNIST data) 
    ```bash
    ## Specify which indices to use among the MNIST -- comparable to DIVA's experiments
    ## change 0 to anything inbtw 0,...,9
    nohup python train.py --model_name="bivae" \
    --latent_dim=128 --hidden_dims 32 64 64 64 --adv_dim 32 32 32 \
    --data_name="multi_rotated_mnist" --angles -45 0 45 --n_styles=3 \
    --selected_inds_fp='/data/hayley-old/Tenanbaum2000/data/Rotated-MNIST/supervised_inds_0.npy' \
    --gpu_id=2
    ```
  
    - E.g.: Train Bivae on multi styles of maptiles from specified cities
    ```bash
    # Train BiVAE on Multi Maptiles MNIST
    nohup python train.py --model_name="bivae" \
    --latent_dim=10 --hidden_dims 32 64 128 256 --adv_dim 32 32 32 --adv_weight 15.0 \
    --data_name="multi_maptiles" \
    --cities la paris \
    --styles CartoVoyagerNoLabels StamenTonerBackground --n_styles=3 \
    --zooms 14 \
    --gpu_id=2 --max_epochs=400   --terminate_on_nan=True  \
    -lr 3e-4 -bs 32 \
    --log_root="/data/hayley-old/Tenanbaum2000/lightning_logs/2021-01-23/" &
    ```
  
### Hyperparameter tuning using `Ray Tune`
- `tune_asha.py`: Use `Tune`'s AsyncHyperBandScheduler to search hyperparameter space more efficiently.
    Use `--tune_metric` to specify the value of `tune.run`'s `metric` argument, e.g. `--tune_metric loss` for 
- `tune_asha_with_beta_scheduler.py`:
- `
    



## Dataset-specific
### Rotated MNIST
- `tune_asha_mnists.py`

### osmnx_roads
- `tune_asha_osmnx_roads.py`

## 