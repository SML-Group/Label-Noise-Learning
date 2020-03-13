# Learning with Label Noise
This code repository is provided for the member of SML-Group led by Prof.Tongliang Liu. Its main topic is Learning with Label Noise. It includes the following: 
- Commonly used datasets and how to generate label noise on synthetic experiments.
- Important baseline.

## Datasets
- Synthetic Datasets: MNIST, [CIFAR10/100](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ), SVHN, Fashion-MNIST.
- Real-world Datasets: Imagenet, Webvision, Clothing1M，Food101.

## How to generate label noise on synthetic experiments
In this section, we consider two kinds of label noise: Class-dependent label-noise and Instance-dependent label-noise.
### Class-dependent label-noise
We corrupted the training and validation sets manually according to true transition matrices T. (See details in utils.py) The flipping setting includes Symmetry Flipping and Pair Flipping. You can use noise rate parameter to control flip rate, use random seed parameter to control different noisy label generation and use split parameter to control the ratio of training and validation set. 
### Instance-dependent label-noise
Coming soon.

## Baseline 
- Cross entropy loss function. It is worth mentioning that PyTorch merged log_softmax and nll_loss to serve as cross entropy loss function. 
- [Forward](https://github.com/giorgiop/loss-correction) and [Backward](https://github.com/giorgiop/loss-correction)
- [Reweight](https://github.com/xiaoboxia/Classification-with-noisy-labels-by-importance-reweighting)
- [T_revision](https://github.com/xiaoboxia/T-Revision)
- [Decoupling](https://github.com/emalach/UpdateByDisagreement)
- [MentorNet](https://github.com/google/mentornet)
- [Co-teaching](https://github.com/bhanML/Co-teaching)
- [Co-teaching Plus](https://github.com/xingruiyu/coteaching_plus)
- [D2L](https://github.com/xingjunm/dimensionality-driven-learning)
- [Symmetric Loss](https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels)
- [Deep Self-Learning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Deep_Self-Learning_From_Noisy_Labels_ICCV_2019_paper.pdf)
- [L_DMI](https://github.com/Newbeeer/L_DMI)
- [Co-Regularization]()
- [DAC](https://github.com/thulas/dac-label-noise)
- [GCE](https://github.com/AlanChou/Truncated-Loss)
- [GLC](https://github.com/mmazeika/glc)



