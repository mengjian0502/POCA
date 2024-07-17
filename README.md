# POCA
Open-sourced Implementation of POCA: Post Training Quantization of Codec Avatar (ECCV, 2024)

Post-Training Quantization for Codec Avatar Model with Jittering-Free rendering. 

## Authors

Jian Meng, Yuecheng Li, Leo (Chenghui) Li, Syed Shakib Sarwar, Diling Wang, and Jae-sun Seo

[[**Website**](https://mengjian0502.github.io/poca.github.io/)] [[**Paper**](https://github.com/mengjian0502/POCA)] [[**Dataset**](https://github.com/facebookresearch/multiface)]

## Overview
<img src="./imgs/overview_camera_ready.png" alt="overview_camera_ready" style="zoom:50%;" />

Although the low precision quantization has been widely investigated, compressing the Codec-Avatar decoder model (e.g., [Deep Appearance Model](https://arxiv.org/pdf/1808.00362), [Pixel Codec Avatar](https://research.facebook.com/publications/pixel-codec-avatars/)) leads to visible and hevaily jittered avatar, which motivated the invention of POCA to compress the decoder **without** introducng additional filtering or finer-grained quantization scheme.

## Usage
### Build the POCA Environment
Following the `requirement.txt` file and install the Conda virtual environment. 

```
conda env create -f requirement.txt
```

### Install Nvidiffrast 
Make sure you manually install the `nvdiffrast (0.3.1)` with the following command:

```
git clone https://github.com/NVlabs/nvdiffrast
pip install .
```

### Start PTQ
#### Download the pre-trained model

The pre-trained full-precision model can be found in the official repo of [MultiFace](https://github.com/facebookresearch/multiface)

By default, the full-precision baseline model should be saved inside the `pretrained_model` folder. 

#### Start POCA

To start the post-training quantization of POCA, please execute the example script file with the ID = 002643814. 

```
bash ptq_002643814.sh
```

The quantized model will be saved inside the path: `./runs/experiment_002643814/PTQ_${arch}_w${wbit}a${wbit}_shape_batch_wise_mask_tau${threshold}_model_calib${model_calib}`

#### Visualize the POCA results

To visualize the output rendered by POCA, execute the following bash file

```
bash visualize_002643814_w8a8_proposed.sh
```

### Cite Us:

```
@inproceedings{poca2024meng,
      author = {Jian Meng and Yuecheng Li and Leo Chenghui Li and Syed Shakib Sarwar and Dilin Wang and Jae-sun Seo},
      title = {POCA: Post-training Quantization with Temporal Alignment for Codec Avatars}, 
      booktitle = {ECCV},
      year = {2024},
}
```

