# NeRF

## Installation

```
cd ./nerf
conda create -n nerf 
conda activate dnerf
pip install -r requirements.txt
```

## Experiment

### Dataset

Download data for three datasets: `lego`, `trex` and `fern`. Place the downloaded dataset according to the following directory structure:
```                                                                                       
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern 
│   │   └── trex
|   ├── nerf_synthetic
|   |   └── lego
```

### Train

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```

---

To train a low-res `trex` NeRF:
```
python run_nerf.py --config configs/trex.txt
```

---

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```

### Test
To test NeRF trained on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt --render_only
```

replace `{DATASET}` with `lego` | `trex` | `fern`.






# D-NeRF

## Installation

```
cd ./D-NeRF
conda create -n dnerf python=3.6
conda activate dnerf
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ..
```

## Experiment

### Dataset

Download data for three datasets: `lego`, `trex` and `standup`. Place the downloaded dataset according to the following directory structure:
```
├── data 
│   ├── lego
│   ├── trex
│   ├── standup 
```

### Train

To test D-NeRF trained on different datasets: 

```
python run_dnerf.py --config configs/{DATASET}.txt 
```

replace `{DATASET}` with `lego` | `trex` | `standup`.

### Test
To test NeRF trained on different datasets: 

```
python run_dnerf.py --config configs/{DATASET}.txt --render_only --render_test
```

replace `{DATASET}` with `lego` | `trex` | `standup`.



# Nerfies

### Dataset

Download data for three datasets: `lego`, `curl` and `shi`. Place the downloaded dataset according to the following directory structure:
```
├── data 
│   ├── lego
│   ├── curl
│   ├── shi
```

## Installation

The code can be run under any environment with Python 3.8 and above.


    conda create --name nerfies python=3.8

Next, install the required packages:

    pip install -r requirements.txt
    
Install the appropriate JAX distribution for your environment by  [following the instructions here](https://github.com/google/jax#installation). For example:

    # For CUDA version 11.0
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
## Training
After preparing a dataset, you can train a Nerfies by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    CUDA_VISIBLE_DEVICES=1 python train_1stage.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/ourdata_lego2.txt

After preparing a dataset, you can  Freeze training a Nerfies steps by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    CUDA_VISIBLE_DEVICES=4 python train_2stage.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        ----gin_configs configs/ourdata_lego2.txt
To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    CUDA_VISIBLE_DEVICES=1 python eval.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin

# Hypernerf

### Dataset

Download data for three datasets: `lego`, `curl` and `shi`. Place the downloaded dataset according to the following directory structure:
```
├── data 
│   ├── lego
│   ├── curl
│   ├── shi
```

## Installation

The code can be run under any environment with Python 3.8 and above.

    conda create --name hypernerf python=3.8

Next, install the required packages:

    pip install -r requirements.txt

Install the appropriate JAX distribution for your environment by  [following the instructions here](https://github.com/google/jax#installation). For example:

    # For CUDA version 11.1
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Training
After preparing a dataset, you can train a Hypernerf by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    CUDA_VISIBLE_DEVICES=4 python train_1stage.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        ----gin_configs configs/ourdata_lego2.txt

After preparing a dataset, you can  Freeze training a Hypernerf steps by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    CUDA_VISIBLE_DEVICES=4 python train_2stage.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        ----gin_configs configs/ourdata_lego2.txt
To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    CUDA_VISIBLE_DEVICES=4 python eval.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        ----gin_configs configs/ourdata_lego2.txt

## Showvideo

You could see our video_show.pptx

## Tool code

In this file,there are some code for data processing. 
the conversion format between colmap and Nerf    
    
    colmap2nerf.py
the conversion format between Nerf and Nerfies and Hypernerf   

    createMeta.py

Data is read and generated in nerf format

    data_preprocess.py

jpg picture format to png picture format

    jpgtopng.py

The conversion between photo and video

    photo_to_video.py
    video)to_pic.py