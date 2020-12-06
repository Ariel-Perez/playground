# Playground

This repository is just a playground for ML related projects.

## Usage
Run benchmarks using the `supervised/benchmark` script.

```bash
python -m supervised.benchmark <dataset> <algorithm>
```

## Supported Datasets
Supported datasets are listed in `supervised/dataset.py`.

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Supported Algorithms
Supported algorithms are listed in `supervised/algorithm.py`.

- Linear Regression
- DNN
- CNN
- [Highway Networks](https://arxiv.org/abs/1507.06228)
