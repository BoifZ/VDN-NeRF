# VDN-NeRF
We propose VDN-NeRF, a method to train neural radiance fields (NeRF) for better geometry under non-Lambertian and dynamic lighting conditions that cause significant variations in the radiance of a point when viewed from different angles.

## [Paper](https://arxiv.org/abs/2303.17968) | [Data](https://drive.google.com/drive/folders/1lw68_ne1ThujwB8uFJ79TA1iAytc3JuZ?usp=sharing)
This is the official repo for the implementation of **VDN-NeRF: Resolving Shape-Radiance Ambiguity via View-Dependence Normalization**.


### Dependencies
  - torch==1.8.0
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.19.2
  - pyhocon==0.3.57
  - icecream==2.1.0
  - tqdm==4.50.2
  - scipy==1.7.0
  - PyMCubes==0.1.2


### Running

- **Training without depth feature**

```shell
python dpt_runner.py --mode train --conf ./confs/womask.conf --case <case_name> -d <image_dir>
```

- **Training with depth feature**

```shell
python dpt_runner.py --mode train --conf ./confs/womask_wdepth.conf --case <case_name> -d <image_dir>
```

- **Extract surface from trained model** 

```shell
python dpt_runner.py --mode validate_mesh --conf <config_file> --case <case_name> -d <image_dir> --is_continue # use latest checkpoint
```

- **Generate depth feature map for finetuning distillation network** 

```shell
python dpt_runner.py --mode getfeats_<epoch> --conf <config_file> --case <case_name> -d <image_dir> # use a specific checkpoint at epoch <epoch>
```

The features projected from SDF network can be found in <dataset_dir>/<case_name>/<image_dir>/depth_from_sdf


### Depth features
- **Extract depth features from input images** 

``` shell
cd wavelet
python predict.py --use_wavelets --normalize_input -ckpt <pre-trained checkpoint folder> -d <image_root> [-full]
```

Extracted features can be found in <image_root>/wavelet_feats[_full]


- **Finetune the distillation network** 

``` shell
cd wavelet
python finetune_for_vdn.py --use_wavelets --normalize_input -ckpt <pre-trained checkpoint folder> -r <dataset_root> --case <case_name> -d <image_dir> -max <feature_max>
```

Here `feature_max` is decided by the distribution of depth_feature_map generated from trained VDN-NeRF, for latter use of normalize all the features to [0, 255]


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{zhu2023vdn,
  title={VDN-NeRF: Resolving Shape-Radiance Ambiguity via View-Dependence Normalization},
  author={Zhu, Bingfan and Yang, Yanchao and Wang, Xulong and Zheng, Youyi and Guibas, Leonidas},
  journal={arXiv preprint arXiv:2303.17968},
  year={2023}
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS) and [WaveletMonoDepth](https://github.com/nianticlabs/wavelet-monodepth). Thanks for these great projects.
