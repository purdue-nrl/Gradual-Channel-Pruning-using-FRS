# Gradual Channel Pruning Using Feature Relevance Scores
This code is related to the paper titled, "Gradual Channel Pruning While Training using Feature Relevance Scores for Convolutional Neural Networks", In arXiv preprint [arXiv:2002.09958, 2020.](https://arxiv.org/pdf/2002.09958.pdf) 

# Available Models
* VGG-16
* ResNet-56, 110, 164 (for CIFAR datasets)
* ResNet-34 (for ImageNet)

# Requirements
* python  == 3.7.4
* PyTorch == 1.4.0
* Numpy   == 1.18.1

# Hyper-parameters
* --model      = model to train
* --dataset    = dataset to train
* --lr         = learning rate
* --batch_size = batch size for training
* --epochs     = total number of training epochs
* --model_dir  = path of a directory to store the trained model

# Additional Hyper-parameters
* --x  = Number of channels that have to be pruned after every few epochs. This value dictates the pruning percentage.
* --n  = Pruning is done after every n epochs.
* --N1 = No pruning is done after N1 epochs till the end of the training allowing the model to converge.

# How to run?
The following are the commands used to gradually prune the model while training using feature relevance scores:

VGG16:
```
python prune_VGG16.py --dataset CIFAR10 --x 160 --n 15 --model_dir './vgg16_cifar10.h5'
(or)
python prune_VGG16.py --dataset CIFAR100 --x 100 --n 15 --model_dir './vgg16_cifar100.h5'
```

ResNet for CIFAR:
```
python prune_resnet.py --dataset CIFAR10 --model resnet-56 --model_dir './res56_cifar10.h5'
(or)
python prune_resnet.py --dataset CIFAR100 --model resnet-110 --model_dir './res110_cifar100.h5'
```
Note that for CIFAR dataset, the available resnet models are resnet-56, resnet-110, resnet-164

ResNet for ImageNet:
```
python prune_imagenet_res34.py --model_dir './res34_imagenet.h5' 
```

# Note
This version of the code doesn't support the pruning of skip connections for the ResNet architectures. I'll update it to include the pruning of skip conncetions soon. 
