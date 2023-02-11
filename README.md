## CSC-Unet: A Novel Convolutional Sparse Coding Strategy based Neural Network for Semantic Segmentation
2023年2月11日12:35:34

The implemented core codes of **CSC-Unet** are open here. 

If you used our CSC-Unet codes, please cite our paper: 

Tang H, He S, Lu X, et al. CSC-Unet: A Novel Convolutional Sparse Coding Strategy based Neural Network for Semantic Segmentation[J]. arXiv preprint arXiv:2108.00408, 2021. https://arxiv.org/abs/2108.00408

If you have any question or collaboration suggestion about our method, please contact wangnizhuan1120@gmail.com. 

The codes of various networks were tested in Pytorch 1.5 version or higher versions(a little bit different from 0.8 version in some functions) in Python 3.8 on Ubuntu machines (may need minor changes on Windows).

### Usage for CSC-Unet

- 1. Clone this repo to local

```
git clone https://github.com/NZWANG/CSC-Unet.git
```

 - 2. Download the experiment dataset from the link below, and put it into the directory: ```./Datasets/CamVid/```	```./Datasets/DeepCrack/```	```./Datasets/Nuclei/```
      1) CamVid: G. J. Brostow, J. Shotton, J. Fauqueur, and R. Cipolla,  “Segmentation and recognition using structure from motion point clouds” http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
      2)  DeepCrack:   Y. Liu, J. Yao, X. Lu, R. Xie, and L. Li,  “DeepCrack: A deep hierarchical feature learning architecture for crack segmentation” https://github.com/yhlleo/DeepCrack
      3)  Nuclei: https://www.kaggle.com/c/data-science-bowl-2018/overview  
- 3. Set the hyper parameters of the experiment in `cfg.py.`

- 4. run the code by command 

```bash
python train.py
```

