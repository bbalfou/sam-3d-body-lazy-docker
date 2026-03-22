# Running Docker Environment

## STEP 1: Create `.env` file 

Inside the `docker` folder create a `.env` file and put inside it you huggingface hub token. This will allow you to download certain models faster.

```sh
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## STEP 2: Create a `checkpoints` folder 
create a checkpoints folder and download the required checkpoints from huggingface hub such that it has the following structure

```
.
└── checkpoints/
    └── sam-3d-body-dinov3/
        ├── assets/
        │   └── mhr_model.pt
        ├── model-config.yaml
        └── model.ckpt
```
## STEP 3: Build Environment

```sh
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
export INSTALL_MOGE=true
export INSTALL_SAM3=true
docker compose -f docker/compose.yaml build --no-cache
docker compose -f docker/compose.yaml up -d
```

or  


```sh
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
docker compose -f docker/compose.yaml build --no-cache
docker compose -f docker/compose.yaml up -d
```


## STEP 4: Run the demo

you can run the normal demo if you like or if your gpu isn't powerful enough you coudl run the lazy version that loads and then each model each step of the pipeline

```sh
python demo_lazy.py \
  --image_folder ./inputs/ \
  --output_folder ./outputs/ \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name vitdet \
  --fov_name moge2 \
  --use_mask
```

```sh
python demo_lazy.py \
  --image_folder ./inputs/ \
  --output_folder ./outputs/ \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name vitdet \
  --use-mask \
  --fov_name moge2 \
  --use_mask
```

`Downloading: "https://github.com/facebookresearch/dinov3/zipball/main" to /home/dev/.cache/torch/hub/main.zip`