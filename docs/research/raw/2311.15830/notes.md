# A-JEPA — Raw Extraction Notes

Sources:
- https://arxiv.org/abs/2311.15830
- https://ar5iv.labs.arxiv.org/html/2311.15830

## Modality adaptation
- Input: mel-spectrogram (128 mel bands, 16kHz).
- Patch embedding: conv kernels 16x16, non-overlapping.
- Built on ViT (image-domain backbone), but with audio-tailored masking.

## Architecture
- Context encoder: 12-layer ViT-B (~86M params).
- Target encoder: same arch, EMA of context encoder.
- Decoder/predictor: 16-layer standard Transformer.

## Curriculum masking
- Anneals from random block masking to time-frequency aware masking near end of training.
- Schedule: f(s) = min(1, sqrt(s * (1 - c0^2) / S) + c0^2), c0 = 0.01.
  s = current step, S = total steps.
- Random block phase: 4 target blocks, scale (0.15, 0.20).
- Time-frequency aware phase: 3 target blocks, scale (0.05, 0.075).
- Motivation: audio spectrograms have high local correlation in time AND frequency,
  so naive image-style masking is too easy.

## Training
- Batch size 512.
- LR 2e-4.
- 24 epochs pretraining on AudioSet-2M.

## Fine-tuning trick
- 10% regularized patch masking on target dataset (not input dropout / zero-fill).
- Modifies self-attention so masked tokens cannot independently form attention weights.

## Headline benchmarks
- AudioSet-2M (AS-2M) mAP: 48.6 (+1.3 vs AudioMAE among audio-only SSL).
- AudioSet-20K (AS-20K) mAP: 38.4.
- ESC-50 acc: 96.3.
- Speech Commands V2 (SPC-2): 98.5.
- Speech Commands V1 (SPC-1): 97.7.
- VoxCeleb (SID): 95.8.

## Takeaway
- Masking design is the high-leverage hyperparameter when porting JEPA to audio.
- Curriculum (image-prior -> audio-aware) > pure audio-aware from scratch.
