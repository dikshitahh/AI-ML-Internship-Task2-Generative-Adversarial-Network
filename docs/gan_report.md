# GAN Implementation Report

## Introduction

Generative Adversarial Networks (GANs) are a class of deep learning models capable of generating realistic synthetic data. A GAN consists of two neural networks—the Generator and the Discriminator—that are trained simultaneously in a competitive manner. The Generator attempts to create realistic images, while the Discriminator learns to distinguish between real and generated images.

## Objective

The objective of this project was to implement a Generative Adversarial Network using PyTorch to generate handwritten digit images from random noise using the MNIST dataset.

## Dataset

The MNIST dataset contains 60,000 grayscale images of handwritten digits. Each image has a resolution of 28 × 28 pixels. Before training, the dataset was normalized to improve training stability.

## Generator Architecture

The Generator converts a 100-dimensional random noise vector into a 28 × 28 grayscale image using multiple fully connected layers followed by ReLU activation functions. A Tanh activation function is applied to the output layer to generate normalized images.

## Discriminator Architecture

The Discriminator receives an image as input and predicts whether it is real or fake. It uses fully connected layers with LeakyReLU activations and a Sigmoid output layer that produces a probability score.

## Training Process

The GAN was trained using an adversarial approach. During each iteration:

- Real images were sampled from the MNIST dataset.
- Fake images were generated from random noise.
- The Discriminator was trained to classify real and fake images correctly.
- The Generator was trained to fool the Discriminator by producing more realistic images.

The training process was repeated for 50 epochs.

## Hyperparameters
--------------------------------------------------
| Hyperparameter     |      Value                |
|--------------------|---------------------------|
| Batch Size         | 128                       |
| Learning Rate      | 0.0002                    |
| Latent Dimension   | 100                       |
| Epochs             | 50                        |
| Optimizer          | Adam                      |
| Loss Function      | Binary Cross Entropy Loss |
--------------------------------------------------

## Challenges Faced

Training GANs is more complex than traditional neural networks because two models are optimized simultaneously. Initially, the generated images appeared noisy, and careful verification of the optimizer configuration and training loop was required to ensure both the Generator and Discriminator were updated correctly. Monitoring generated samples throughout training helped evaluate learning progress.

## Results

Generated images were saved every five epochs to observe the Generator's improvement over time. Early epochs produced noisy images, while later epochs showed more recognizable handwritten digit patterns. The trained Generator and Discriminator models were also saved for future use.

## Conclusion

This project successfully implemented a basic GAN for handwritten digit generation. It provided practical experience with adversarial learning, generator-discriminator training, and synthetic image generation using PyTorch.
