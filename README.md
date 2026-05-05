# Fine-Tuning ASR Models for Child Speech

## Project Description

I built a child-speech ASR fine-tuning pipeline that compares Whisper and Wav2Vec2 on the OGI Kids dataset, analyzing how model size, fine-tuning, hyperparameters, training data size, age-group transfer, and augmentation affect word error rate.

## Tech Stack

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Whisper
- Wav2Vec2
- OGI Kids Speech Corpus
- Matplotlib
- NumPy / Pandas

## Overview

This project explores how pretrained automatic speech recognition models perform on children’s speech, which is often harder to recognize than adult speech because of differences in pitch, pronunciation, speaking rate, and variability. The project evaluates both zero-shot and fine-tuned performance for Whisper and Wav2Vec2, then studies which training choices most affect word error rate.

## Models Evaluated

- Whisper-tiny.en
- Whisper-base.en
- Wav2Vec2-base-960h

## Experiments

The project includes several experiments:

1. Compared Whisper-tiny and Whisper-base in zero-shot and fine-tuned settings.
2. Compared Wav2Vec2 zero-shot performance against fine-tuned performance.
3. Swept hyperparameters including learning rate, batch size, and training steps.
4. Tested the impact of training-set size on model performance.
5. Analyzed age-group transfer across children ages 4–14.
6. Tested audio augmentation strategies including noise, gain scaling, speed perturbation, and VTLP.

## Key Results

- Whisper-base performed best overall in the main comparison.
- Wav2Vec2 improved substantially after fine-tuning, showing strong benefit from domain adaptation.
- Younger children, especially ages 4–6, were the hardest group for both models.
- Training on all age groups gave the best cross-age generalization.
- Simple noise and gain augmentation hurt performance, while VTLP provided only limited improvement.

## Future Work

Future improvements could include repeated-seed experiments, broader hyperparameter sweeps, more targeted child-speech augmentation, and deeper error analysis across substitutions, deletions, and insertions.
