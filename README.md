# Chainer implementation of DeepFill v1 and v2

[DeepFillv1 Paper](https://arxiv.org/abs/1801.07892) | [Official Implementation](https://github.com/JiahuiYu/generative_inpainting) 

[DeepFillv2 Paper](https://arxiv.org/abs/1806.03589)

## Requirements

```
numpy
opencv_python
chainer >= 6.0.0b
Pillow
PyYAML
```

## Datasets
Please save text files that contain paths to images in distinct lines, and specify them for `IMAGE_FLIST` in the config file (`src/contextual_attention.yml` and `src/gated_convolution.yml`).

```angular2
IMAGE_FLIST: [
  'paths_for_training_image.txt', # for training
  'paths_for_validation_image.txt', # for validation
]
```

When you train DeepFillv2 with edge image input, please save edge image in advance, and specify the paths to the text files that contain edge images paths for `EDGE_FLIST` in the config file (`src/gated_convolution.yml`). The orders of image paths and edge paths must be the same.

```angular2
EDGE_FLIST: [
  'paths_for_training_edge.txt', # for training
  'paths_for_validation_edge.txt', # for validation
]
```
You can train without edge input if you do not specify anything for `EDGE_FLIST`.


Edge image example:

![edge image](https://drive.google.com/uc?export=view&id=18elb8ybskoDNeVTViNHOqQ91i5pvEfbj "edge image")

Background and edge values should be 0 and 255 respectively.

## Training
Only single GPU training is supported.

- DeepFillv1
  - Modify `contextual_attention.yml` to set `IMAGE_FLIST`, `MODEL_RESTORE`,`EVAL_FOLDER` and other parameters.
  - Run 
  ```
  cd src
  python train_contextual_attention.py
  ```

- DeepFillv2
  - Modify `gated_convolution.yml` to set `IMAGE_FLIST`, `EDGE_FLIST`, `MODEL_RESTORE`,`EVAL_FOLDER` and other parameters.
  - Run 
  
  ```
  cd src
  python train_gated_convolution.py
  ```

## Validation
Run
```angular2
python test.py --model [v1 or v2] --config_path [path to config] --snapshot [path to snapshot] --name [file name to save]
```

## Results on ImageNet
- DeepFillv1 (top: original, middle: input, bottom: output)
![contextual_attention](https://drive.google.com/uc?export=view&id=1dMmu2e_z8bKZ6HhA8c9jYOOSruxSw63v "contextual_attention")
- DeepFillv2 with edge input (top: original, middle: input, bottom: output)
![gated_convolution](https://drive.google.com/uc?export=view&id=1LzsOuT7ZWNocBw9EHYII0mDS676YPyef "gated_convolution")


## Citing
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```