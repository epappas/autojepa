# CNN-JEPA — Abstract

Source: https://arxiv.org/abs/2408.07514
Fetched: 2026-05-15

> Self-supervised learning (SSL) has become an important approach in
> pretraining large neural networks, enabling unprecedented scaling of model
> and dataset sizes. While recent advances like I-JEPA have shown promising
> results for Vision Transformers, adapting such methods to Convolutional
> Neural Networks (CNNs) presents unique challenges. In this paper, we
> introduce CNN-JEPA, a novel SSL method that successfully applies the joint
> embedding predictive architecture approach to CNNs. Our method incorporates
> a sparse CNN encoder to handle masked inputs, a fully convolutional
> predictor using depthwise separable convolutions, and an improved masking
> strategy. Using the ImageNet-100 dataset, we show that CNN-JEPA outperforms
> I-JEPA with ViT architectures and other leading SSL methods (BYOL, SimCLR,
> VICReg) in terms of linear top-1 accuracy, while requiring shorter training
> times. Our approach offers a simpler, more efficient alternative to existing
> SSL methods for CNNs, requiring minimal augmentations and no separate
> projector network.
