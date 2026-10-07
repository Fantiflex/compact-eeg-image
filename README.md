# Compact EEG-to-Image Retrieval

Research project for Track 1 of the EEG/EMG Foundation Challenge 2026.

## Goal

Investigate whether a compact EEG encoder can benefit from self-supervised
pretraining for EEG-to-image retrieval and approach the performance of
large EEG foundation models.

## Research questions

1. Does masked EEG pretraining improve image retrieval compared with
   training the same architecture from scratch?
2. How much performance can a compact EEG model recover relative to REVE?
3. How does downstream performance scale with the amount of EEG pretraining data?

## Planned baselines

- EEGNet
- REVE
- CompactEEG trained from scratch
- CompactEEG with self-supervised pretraining

## Challenge

Track 1: EEG-to-Image Retrieval

Input:

    EEG epoch (B, C, T)

Output:

    image embedding (B, D)

Evaluation:

    cosine-similarity image retrieval