# Consistent-Explanations-by-Contrastive-Learning
Official PyTorch code for CVPR 2022 paper - [Consistent Explanations by Contrastive Learning][1]


Post-hoc explanation methods, e.g., Grad-CAM, enable humans to inspect the spatial regions responsible for a particular network decision. However, it is shown that such explanations are not always consistent with human priors, such as consistency across image transformations. Given an interpretation algorithm, e.g., Grad-CAM, we introduce a novel training method to train the model to produce more consistent explanations. Since obtaining the ground truth for a desired model interpretation is not a well-defined task, we adopt ideas from contrastive self-supervised learning, and apply them to the interpretations of the model rather than its embeddings. We show that our method, Contrastive Grad-CAM Consistency (CGC), results in Grad-CAM interpretation heatmaps that are more consistent with human annotations while still achieving comparable classification accuracy. Moreover, our method acts as a regularizer and improves the accuracy on limited-data, fine-grained classification settings. In addition, because our method does not rely on annotations, it allows for the incorporation of unlabeled data into training, which enables better generalization of the model.

![Teaser image][teaser]

<br/>

## Bibtex
```
@InProceedings{Pillai_2022_CVPR,
author = {Pillai, Vipin and Abbasi Koohpayegani, Soroush and Ouligian, Ashley and Fong, Dennis and Pirsiavash, Hamed},
title = {Consistent Explanations by Contrastive Learning},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}
```

## Pre-requisites
- Pytorch 1.3 - Please install [PyTorch](https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed. 

## Datasets
 - [ImageNet - 1K](https://www.image-net.org/download.php)
 - [CUB-200](https://vision.cornell.edu/se3/caltech-ucsd-birds-200/)
 - [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
 - [Stanford Cars-196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
 - [VGG Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Training

#### Train and evaluate a ResNet50 model on the ImageNet dataset using our CGC loss
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_eval_cgc.py /datasets/imagenet -a resnet50 -p 100 -j 8 -b 256 --lr 0.1 --lambda 0.5 -t 0.5 --save_dir <SAVE_DIR> --log_dir <LOG_DIR>
```

#### Train and evaluate a ResNet50 model on 1pc labeled subset of ImageNet dataset and the rest as unlabeled dataset. We initialize the model from SwAV
For the below command, <PATH_TO_SWAV_MODEL_PRETRAINED> can be downloaded from the github directory of SwAV - https://github.com/facebookresearch/swav
We use the model checkpoint listed on the first row (800 epochs, 75.3% ImageNet top-1 acc.) of the Model Zoo of the above repository.

```
CUDA_VISIBLE_DEVICES=0,1 python train_imagenet_1pc_swav_cgc_unlabeled.py <PATH_TO_1%_IMAGENET> -a resnet50 -b 128 -j 8 --lambda 0.25 -t 0.5 --epochs 50 --lr 0.02 --lr_last_layer 5 --resume <PATH_TO_SWAV_MODEL_PRETRAINED> --save_dir <SAVE_DIR> --log_dir <LOG_DIR> 2>&1 | tee <PATH_TO_CMD_LOG_FILE>
```

<br/>

## Checkpoints
* ResNet50 model pre-trained on ImageNet - [link](https://drive.google.com/drive/folders/1n7lFew0CdWuYCpR1kImMt7UC7_vsO5CT?usp=sharing)

## Evaluation

#### Evaluate model checkpoint using Content Heatmap (CH) evaluation metric
We use the evaluation code adapted from the TorchRay framework.
* Change directory to TorchRay and install the library. Please refer to the [TorchRay](https://github.com/facebookresearch/TorchRay) repository for full documentation and instructions.
    * cd TorchRay
    * python setup.py install

* Change directory to TorchRay/torchray/benchmark
    * cd torchray/benchmark

* For the ImageNet & CUB-200 datasets, this evaluation requires the following structure for validation images and bounding box xml annotations
    * <PATH_TO_FLAT_VAL_IMAGES_BBOX>/val/*.JPEG - Flat list of validation images
    * <PATH_TO_FLAT_VAL_IMAGES_BBOX>/annotation/*.xml - Flat list of annotation xml files

##### Evaluate ResNet50 models trained on the full ImageNet dataset
```
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_energy_inside_bbox.py <PATH_TO_FLAT_VAL_IMAGES_BBOX> -j 0 -b 1 --resume <PATH_TO_SAVED_CHECKPOINT_FILE> --input_resize 448 -a resnet50
```

##### Evaluate ResNet50 models trained on the CUB-200 fine-grained dataset
```
CUDA_VISIBLE_DEVICES=0 python evaluate_finegrained_gradcam_energy_inside_bbox.py <PATH_TO_FLAT_VAL_IMAGES_BBOX> --dataset cub -j 0 -b 1 --resume <PATH_TO_SAVED_CHECKPOINT_FILE> --input_resize 448 -a resnet50
```

##### Evaluate ResNet50 models trained from SwAV initialized models with 1pc labeled subset of ImageNet and rest as unlabeled
```
CUDA_VISIBLE_DEVICES=0 python evaluate_swav_imagenet_gradcam_energy_inside_bbox.py <PATH_TO_IMAGENET_VAL_FLAT> -j 0 -b 1 --resume <PATH_TO_SAVED_CHECKPOINT_FILE> --input_resize 448 -a resnet50
```

<br/>

#### Evaluate model checkpoint using Insertion AUC (IAUC) evaluation metric
Change to directory RISE/ and follow the below commands:

##### Evaluate pre-trained ResNet50 model
```
CUDA_VISIBLE_DEVICES=0 python evaluate_auc_metrics.py --pretrained
```

##### Evaluate ResNet50 model trained using our CGC method
```
CUDA_VISIBLE_DEVICES=0 python evaluate_auc_metrics.py --ckpt-path <PATH_TO_SAVED_CHECKPOINT_FILE>
```

<br/>

## License
This project is licensed under the MIT License.

[1]: https://arxiv.org/pdf/2110.00527.pdf
[teaser]: https://github.com/UMBCvision/Consistent-Explanations-by-Contrastive-Learning/blob/main/misc/teaser_image.png
