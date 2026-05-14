# Abstract — On the Importance of Hyperparameters and Data Augmentation for SSL

**Source URL:** https://arxiv.org/abs/2207.07875
**Fetched:** 2026-05-15

> Self-Supervised Learning (SSL) has become a very active area of Deep Learning
> research where it is heavily used as a pre-training method for classification
> and other tasks. However, the rapid pace of advancements in this area comes
> at a price: training pipelines vary significantly across papers, which
> presents a potentially crucial confounding factor. Here, we show that,
> indeed, the choice of hyperparameters and data augmentation strategies can
> have a dramatic impact on performance. To shed light on these neglected
> factors and help maximize the power of SSL, we hyperparameterize these
> components and optimize them with Bayesian optimization, showing improvements
> across multiple datasets for the SimSiam SSL approach. Realizing the
> importance of data augmentations for SSL, we also introduce a new automated
> data augmentation algorithm, GroupAugment, which considers groups of
> augmentations and optimizes the sampling across groups. In contrast to
> algorithms designed for supervised learning, GroupAugment achieved
> consistently high linear evaluation accuracy across all datasets we
> considered. Overall, our results indicate the importance and likely
> underestimated role of data augmentation for SSL.
