SRGAN-PyTorch
============================
Introduction
----------------------------
This project is a Pytorch implementation of SRGAN based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802). We implement data parallel, distributed data parallel, and lighting in HPC server and compared the PSNR\/SSIM of the results of Bicubic interpolation, SRResNet, and SRGAN on RTX8000 with one or multipile GPUs.

Requirement
----------------------------
* Argparse
* Numpy
* Pillow
* Python 3.7
* PyTorch
* TorchVision
* tqdm


Usage
----------------------------
### Model
The SRGAN model is implemented in model.py, including ResidualBlock, Generator, Discriminator.

### Processing of datasets
utils.py is used to pre-process dataset image from DIV2K, cropping into training_set, validation_set, and adding FeatureExtractor

### Training

Download the data to the ./data/ folder. The [pretrained weight of SRResNet](https://drive.google.com/file/d/126GzYaRBprQYju1g0WGVF_5UvbMIYkh3/view?usp=sharing) is optional
Run the script train.py
```
$ python train_time.py --trainset_dir $TRAINDIR --validset_dir $VALIDDIR --upscale_factor 4 --pretrain SRResNet_weight.pth --cuda

usage: train.py [-h] [--trainset_dir TRAINSET_DIR]
                [--validset_dir VALIDSET_DIR] [--upscale_factor {2,4,8}]
                [--epochs EPOCHS] [--resume RESUME]
                [--mode {adversarial,generator}] [--pretrain PRETRAIN]
                [--cuda] [--dp] [--ddp]

optional arguments:
  -h, --help            show this help message and exit
  --trainset_dir TRAINSET_DIR
                        training dataset path
  --validset_dir VALIDSET_DIR
                        validation dataset path
  --upscale_factor {2,4,8}
                        super resolution upscale factor
  --epochs EPOCHS       training epoch number
  --resume RESUME       continues from epoch number
  --mode {adversarial,generator}
                        apply adversarial training
  --pretrain PRETRAIN   load pretrained generator model
  --cuda                Using GPU to train
  --dp                  Using data parallel
  --ddp                 Using distributed data parallel
```

### Testing

Download the [SRGAN weight](https://drive.google.com/file/d/1dsa67sCyM29_Tor124KjP2vVYxO8sXD3/view?usp=sharing) that was trained with DIV2K dataset.

Run the script test_image.py

```
$ python test_image.py --image $IMG --weight $WEIGHT --cuda

usage: test_image.py [-h] [--image IMAGE] [--upscale_factor {2,4,8}]
                     [--weight WEIGHT] [--downsample {None,bicubic}] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         input image
  --upscale_factor {2,4,8}
                        super resolution upscale factor
  --weight WEIGHT       generator weight file
  --downsample {None,bicubic}
                        Downsample the input image before applying SR
  --cuda                Using GPU to run
```

### Crop image

To visualize and compare the detail in the image, this script to save multiple patches from input image with colored bounding box. The cropped images will be saved in the same directory as input image. When the saved coordinates is not specified, the program will prompt image for used to select bounding box from image. Then the coordinates will be saved to crop other images.


```
$ python get_img_crop.py --image $IMG

usage: get_img_crop.py [-h] [--image IMAGE] [--coords COORDS]

optional arguments:
  -h, --help       show this help message and exit
  --image IMAGE    input image to be croped
  --coords COORDS  Loading the bounding box coordinates from saved file.
                   Manual selecting boxes when no saved coordinates is used.
```

Sample Results
----------------------------
### Sample from DIV2K validation set  

#### Bicubic
![sample1_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample1_lr.png "Bicubic")   

#### SRGAN
![sample1_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample1_sr.png "SRGAN")   

#### Bicubic
![sample2_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample2_lr.png "Bicubic")   

#### SRGAN
![sample2_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample2_sr.png "SRGAN")    

#### Bicubic
![sample3_lr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample3_lr.png "Bicubic")   

#### SRGAN
![sample3_sr](https://github.com/Maggiking/SRGAN-PyTorch/blob/master/images/sample3_sr.png "SRGAN")   

### PSNR and SSIM(luma(Y) channels)
![image](https://user-images.githubusercontent.com/58716946/208552766-35d1867e-bd60-4313-98ee-8fcb2253ec80.png)

### PSNR and SSIM(3YCrCb channels)
![image](https://user-images.githubusercontent.com/58716946/208552824-8f8c1c34-7e5f-4b2e-9f09-98da3b618152.png)


### Test image   

#### Low Res | SRGAN | Ground Truth
![image](https://user-images.githubusercontent.com/58716946/208553045-5c9d8286-dca1-4091-b244-3a09b6a0f2a7.png)

#### Low Res | SRGAN | Ground Truth
![image](https://user-images.githubusercontent.com/58716946/208553111-7b22c473-3c03-48be-8ca0-24c6203ddbbb.png)

Observation and Results
----------------------------
1.SRGAN needs to be trained for more epochs (>10,000).
2.Training devices should not affect loss convergence.
3.PSNR and SSIM do not reflect perceived image quality, image quality is subjective.
4.DDP is significantly faster than DP for single node multi-GPU training. It also allows training across multiple nodes.
5.Pytorch Lightning offers even better scaling efficiency. Lightning w/ 2 GPUs is as fast as DDP w/ 4 GPUs.
6.Scaling efficiency depends on optimization. Lightning is highly optimized for distributed training, also works well with SLURM.
7.Performance difference between our DDP and Lightning may be partially caused by different communication backend. We used GLOO and Lightning uses NCCL which may be more suitable for the HPC clusters.




