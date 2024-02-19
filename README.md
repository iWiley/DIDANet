# DIDANet: An Attentive Neural Decision Framework
### [website](didanet.wistu.cn) 'https://didanet.wistu.cn'

#### We present a multimodal deep imaging decision attention network (DIDANet) that integrates radiomics with traditional ML modules and introduces them into a neural network. DIDANet effectively predicts ER and postoperative TACE benefits in HCC patients. Additionally, it addresses overfitting in small-sample medical models to some extent.

## Network Architecture
![输入图片说明](photos/4.png)
## Quantitative & Qualitative Results on CDD and WHU
![输入图片说明](photos/5.png)
![输入图片说明](photos/2.png)
##  Usage
### Requirements
```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2
```
Please see ```requirements.txt``` for all the other requirements.
### Installation
Clone this repo:
```
git clone https://github.com/LiuFxxx/T-GDCD.git
```
### Setting up conda environment:
Create a virtual ```conda``` environment named ```GDCD``` with the following command:
```
conda create --name GDCD --file requirements.txt
conda activate GDCD
```

