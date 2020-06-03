# SRZoo
<p align="center">
  <img src="figures/logo.png" alt="SRZoo">
</p>
<p align="center">
  <a href="https://github.com/idearibosome/srzoo">
    <img src="https://img.shields.io/badge/srzoo-supported-brightgreen" alt="SRZoo" />
  </a>
  <a href="#pre-trained-super-resolution-models">
    <img src="https://img.shields.io/badge/models-29-blue" />
  </a>
</p>


## Introduction
SRZoo is a collection of toolkits and models for deep learning-based image super-resolution.
It provides various pre-trained state-of-the-art super-resolution models that are ready for use.

Here are the key features of SRZoo:
- SRZoo provides **official** pre-trained models of various super-resolution methods.
- With SRZoo, you can easily obtain the super-resolved images from the supported super-resolution methods.
- It is possible to employ the super-resolution models in various environments such as GPUs supporting CUDA and web browsers via TensorFlow.js.
- It is possible to compare the performance of the super-resolution methods with the same evaluation metrics and the same environment.

You can find our motivation and some detailed description of SRZoo such as performance comparison in the following paper.
- J.-H. Choi, J.-H. Kim, J.-S. Lee. SRZoo: an integrated repository for super-resolution using deep learning. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), May 2020 **[[Paper]](https://doi.org/10.1109/ICASSP40776.2020.9054533)** **[[arXiv]](https://arxiv.org/abs/2006.01339)**

## Requirements

- Python 3.6 or newer
- TensorFlow 1.12 or newer


## Pre-trained super-resolution models

We currently provide the following pre-trained super-resolution models, where the model parameters are provided by the original authors.
Please cite the paper of the original authors when you use the models.

| Name | Config | Upscaling factor | Model | Source |
| --- | --- | --- | --- | --- |
| EDSR-baseline | [edsr_baseline.json](configs/edsr_baseline.json) | 2 | [edsr_baseline_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_baseline_x2.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EDSR-baseline | [edsr_baseline.json](configs/edsr_baseline.json) | 3 | [edsr_baseline_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_baseline_x3.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EDSR-baseline | [edsr_baseline.json](configs/edsr_baseline.json) | 4 | [edsr_baseline_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_baseline_x4.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EDSR | [edsr.json](configs/edsr.json) | 2 | [edsr_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_x2.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EDSR | [edsr.json](configs/edsr.json) | 3 | [edsr_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_x3.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EDSR | [edsr.json](configs/edsr.json) | 4 | [edsr_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/edsr_x4.pb) | [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) |
| EUSR | [eusr.json](configs/eusr.json) | 2 | [eusr_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/eusr_x2.pb) | [EUSR-TensorFlow](https://github.com/junhyukk/EUSR-Tensorflow) |
| EUSR | [eusr.json](configs/eusr.json) | 4 | [eusr_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/eusr_x4.pb) | [EUSR-TensorFlow](https://github.com/junhyukk/EUSR-Tensorflow) |
| EUSR | [eusr.json](configs/eusr.json) | 8 | [eusr_x8.pb](http://mcml.yonsei.ac.kr/files/srzoo/eusr_x8.pb) | [EUSR-TensorFlow](https://github.com/junhyukk/EUSR-Tensorflow) |
| DBPN | [dbpn.json](configs/dbpn.json) | 2 | [dbpn_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/dbpn_x2.pb) | [DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch) |
| DBPN | [dbpn.json](configs/dbpn.json) | 4 | [dbpn_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/dbpn_x4.pb) | [DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch) |
| DBPN | [dbpn.json](configs/dbpn.json) | 8 | [dbpn_x8.pb](http://mcml.yonsei.ac.kr/files/srzoo/dbpn_x8.pb) | [DBPN-Pytorch](https://github.com/alterzero/DBPN-Pytorch) |
| RCAN | [rcan.json](configs/rcan.json) | 2 | [rcan_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/rcan_x2.pb) | [RCAN](https://github.com/yulunzhang/RCAN) |
| RCAN | [rcan.json](configs/rcan.json) | 3 | [rcan_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/rcan_x3.pb) | [RCAN](https://github.com/yulunzhang/RCAN) |
| RCAN | [rcan.json](configs/rcan.json) | 4 | [rcan_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/rcan_x4.pb) | [RCAN](https://github.com/yulunzhang/RCAN) |
| RCAN | [rcan.json](configs/rcan.json) | 8 | [rcan_x8.pb](http://mcml.yonsei.ac.kr/files/srzoo/rcan_x8.pb) | [RCAN](https://github.com/yulunzhang/RCAN) |
| MSRN | [msrn.json](configs/msrn.json) | 2 | [msrn_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/msrn_x2.pb) | [MSRN-PyTorch](https://github.com/MIVRC/MSRN-PyTorch) |
| MSRN | [msrn.json](configs/msrn.json) | 3 | [msrn_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/msrn_x3.pb) | [MSRN-PyTorch](https://github.com/MIVRC/MSRN-PyTorch) |
| MSRN | [msrn.json](configs/msrn.json) | 4 | [msrn_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/msrn_x4.pb) | [MSRN-PyTorch](https://github.com/MIVRC/MSRN-PyTorch) |
| 4PP-EUSR | [4pp_eusr.json](configs/4pp_eusr.json) | 4 | [4pp_eusr_pirm_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/4pp_eusr_pirm_x4.pb) | [tf-perceptual-eusr](https://github.com/idearibosome/tf-perceptual-eusr) |
| ESRGAN | [esrgan.json](configs/esrgan.json) | 4 | [esrgan_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/esrgan_x4.pb) | [ESRGAN](https://github.com/xinntao/ESRGAN) |
| RRDB | [rrdb.json](configs/rrdb.json) | 4 | [rrdb_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/rrdb_x4.pb) | [ESRGAN](https://github.com/xinntao/ESRGAN) |
| CARN | [carn.json](configs/carn.json) | 2 | [carn_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/carn_x2.pb) | [CARN-pytorch](https://github.com/nmhkahn/CARN-pytorch) |
| CARN | [carn.json](configs/carn.json) | 3 | [carn_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/carn_x3.pb) | [CARN-pytorch](https://github.com/nmhkahn/CARN-pytorch) |
| CARN | [carn.json](configs/carn.json) | 4 | [carn_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/carn_x4.pb) | [CARN-pytorch](https://github.com/nmhkahn/CARN-pytorch) |
| FRSR | [frsr_x2.json](configs/frsr_x2.json) | 2 | [frsr_x2.pb](http://mcml.yonsei.ac.kr/files/srzoo/frsr_x2.pb) | [NatSR](https://github.com/JWSoh/NatSR) |
| FRSR | [natsr.json](configs/natsr.json) | 3 | [frsr_x3.pb](http://mcml.yonsei.ac.kr/files/srzoo/frsr_x3.pb) | [NatSR](https://github.com/JWSoh/NatSR) |
| FRSR | [natsr.json](configs/natsr.json) | 4 | [frsr_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/frsr_x4.pb) | [NatSR](https://github.com/JWSoh/NatSR) |
| NatSR | [natsr.json](configs/natsr.json) | 4 | [natsr_x4.pb](http://mcml.yonsei.ac.kr/files/srzoo/natsr_x4.pb) | [NatSR](https://github.com/JWSoh/NatSR) |


## Super-resolved image retrieval

SRZoo offers a simple image retrieval via ```get_sr.py```, e.g.,
```
python get_sr.py --config_path=configs/edsr.json --model_path=edsr_x4.pb --input_path=LR --output_path=SR --scale=4
```

Arguments:
- ```config_path```: Path of the model config file.
- ```model_path```: Path of the pre-trained model file.
- ```input_path```: Path of the input low-resolution images.
- ```output_path```: Path of the output super-resolved images will be saved.
- ```scale```: Upscaling factor.
- ```self_ensemble```: Specify this to employ [geometric self-ensemble](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf).
- ```cuda_device```: CUDA device index to be used (will be set to the environment variable 'CUDA_VISIBLE_DEVICES').

※ Some models can be run only on GPUs due to the different ordering of the dimensions.

## Performance evaluation

With the obtained super-resolved images, it is possible to evaluate the performance via ```evaluate_sr.py```, e.g.,
```
python evaluate_sr.py --sr_path=SR --truth_path=HR
```

Arguments:
- ```sr_path```: Path of the super-resolved images.
- ```truth_path```: Path of the ground-truth images.
- ```shave_borders```: The amount of shaving pixles on borders of the images. It is usually set to the upscaling factor.
- ```color_mode```: Color conversion mode. ycbcry: Y channel of the YCbCr color space. rgb: RGB channels of the RGB color space.
- ```evaluators```: Comma-separated evaluation methods. The evaluators in the [```evaluators/```](evaluators/) folder will be used.
- ```ouptut_name```: Filename of the output CSV file.

You can also write your own evaluation metric by implementing an evaluator class that inherits ```BaseEvaluator``` in the ```evaluators/``` folder.


## Model conversion

It is possible to convert the other pre-trained super-resolution models.
Please refer to the [```converter/```](converter/) folder for more information.
In addition, please refer to the [```config/```](config/) folder to write your own model config file along with the converted model.


## Miscellaneous

### Image downscaling utilities

We also provide the downscaling utilities for evaluating the super-resolution models, which are in the [```utils/downscale/```](utils/downscale/) folder.

### Employing other image processing models

Since SRZoo is developed to deal with models considering images as both inputs and outputs, our repository can be used to employ the other image processing algorithms with only a few modifications.
As a proof-of-concept, we provide a pre-trained [deep learning-based image compression model](https://github.com/fab-jul/imgcomp-cvpr) in SRZoo.

| Name | Config | Model | Source |
| --- | --- | --- | --- |
| fab-jul/imgcomp-cvpr | [fabjul_imgcomp.json](configs/fabjul_imgcomp.json) | [fabjul_imgcomp_a.pb](http://mcml.yonsei.ac.kr/files/srzoo/fabjul_imgcomp_a.pb) | [imgcomp-cvpr](https://github.com/fab-jul/imgcomp-cvpr) |
| fab-jul/imgcomp-cvpr | [fabjul_imgcomp.json](configs/fabjul_imgcomp.json) | [fabjul_imgcomp_b.pb](http://mcml.yonsei.ac.kr/files/srzoo/fabjul_imgcomp_b.pb) | [imgcomp-cvpr](https://github.com/fab-jul/imgcomp-cvpr) |
| fab-jul/imgcomp-cvpr | [fabjul_imgcomp.json](configs/fabjul_imgcomp.json) | [fabjul_imgcomp_c.pb](http://mcml.yonsei.ac.kr/files/srzoo/fabjul_imgcomp_c.pb) | [imgcomp-cvpr](https://github.com/fab-jul/imgcomp-cvpr) |

To use these models, simply set the upscaling factor (e.g., ```--scale``` option of ```get_sr.py```) to 1.
Note that only the GPU mode is currently supported for these models.
