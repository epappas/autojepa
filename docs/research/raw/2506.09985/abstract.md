# V-JEPA 2 — Abstract

Source: https://arxiv.org/abs/2506.09985
Fetched: 2026-05-15

> A major challenge for modern AI is to learn to understand the world and
> learn to act largely by observation. This paper explores a self-supervised
> approach that combines internet-scale video data with a small amount of
> interaction data (robot trajectories), to develop models capable of
> understanding, predicting, and planning in the physical world. We first
> pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2,
> on a video and image dataset comprising over 1 million hours of internet
> video. V-JEPA 2 achieves strong performance on motion understanding (77.3
> top-1 accuracy on Something-Something v2) and state-of-the-art performance
> on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100). After
> aligning V-JEPA 2 with a large language model, we demonstrate
> state-of-the-art performance on multiple video question-answering tasks at
> the 8 billion parameter scale. Finally, we show how self-supervised learning
> can be applied to robotic planning tasks by post-training a latent
> action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of
> unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC
> zero-shot on Franka arms in two different labs and enable picking and
> placing of objects using planning with image goals.
