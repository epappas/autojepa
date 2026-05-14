# JEPA Audio Design Choices — Abstract

Source: https://arxiv.org/abs/2405.08679
Fetched: 2026-05-15

> This paper addresses the problem of self-supervised general-purpose audio
> representation learning. We explore the use of Joint-Embedding Predictive
> Architectures (JEPA) for this task, which consists of splitting an input
> mel-spectrogram into two parts (context and target), computing neural
> representations for each, and training the neural network to predict the
> target representations from the context representations. We investigate
> several design choices within this framework and study their influence
> through extensive experiments by evaluating our models on various
> downstream tasks. We show that this approach can be effectively applied to
> audio representation learning, and that some effective design choices in
> the image domain lead to poor performance on audio. In particular, we
> demonstrate the importance of the choice of context and target portions of
> the mel-spectrogram, which significantly impacts the model's quality.
