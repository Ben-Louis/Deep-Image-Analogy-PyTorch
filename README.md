# Deep-Image-Analogy

[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

This project is a python implementation of [Deep Image Analogy](https://arxiv.org/abs/1705.01088).

## Some results

![](results/results.jpg)

## Requirements

 - python3

 - opencv3

   If you use anaconda, you can install opencv3 by  ```conda install opencv```

 - [pytorch](http://pytorch.org/) >= 1.7

 - [pycuda](https://pypi.org/project/pycuda/) (optional)

## Usage (demo)

#### Use GPU
```bash
python main.py --use_cuda
```

#### CPU only
```bash
python main.py
```
I train a series of invert blocks which estimate the inputs given the outputs for each stage of Vgg19. 
The results are slightly worse than optimization-based method.

## Acknowledgments

My project acknowledge the official code [Deep-Image-Analogy](https://github.com/msracver/Deep-Image-Analogy), [pytorch](http://pytorch.org/), and [another pytorch implementation](https://github.com/harveyslash/Deep-Image-Analogy-PyTorch). Especially, thanks for the authors of this amazing algorithm.