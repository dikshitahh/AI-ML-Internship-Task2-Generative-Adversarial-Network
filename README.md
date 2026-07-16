# AI-ML-Internship-Task2-Generative-Adversarial-Network

# Generative Adversarial Network (GAN) for MNIST Image Generation

## Project Overview

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate synthetic handwritten digit images based on the MNIST dataset. A GAN consists of two neural networks—the Generator and the Discriminator—that compete against each other during training. The Generator creates fake images from random noise, while the Discriminator attempts to distinguish between real and generated images. Through adversarial training, the Generator gradually learns to produce realistic handwritten digits.

## Dataset

- Dataset:MNIST Handwritten Digits
- Training Images: 60,000
- Image Size: 28 × 28 pixels
- Channels: 1 (Grayscale)
- Classes: 10 (Digits 0–9)

The images were normalized to the range [-1, 1] before training.

## GAN Architecture

### Generator

The Generator receives a 100-dimensional random noise vector (latent vector) and transforms it into a synthetic 28×28 grayscale image using fully connected layers with ReLU activation functions. The final layer uses a Tanh activation function.

### Discriminator

The Discriminator receives either a real or generated image and predicts whether it is real or fake. It consists of fully connected layers with LeakyReLU activations and a Sigmoid output layer that returns a probability.

## Hyperparameters
----------------------------------------------------------
| Parameter        | Value                               |
|------------------|-------------------------------------|
| Batch Size       | 128                                 |
| Learning Rate    | 0.0002                              |
| Latent Dimension | 100                                 |
| Epochs           | 50                                  |
| Optimizer        | Adam                                |
| Loss Function    | Binary Cross Entropy Loss (BCELoss) |
----------------------------------------------------------

## Training Process

1. Load and preprocess the MNIST dataset.
2. Generate fake images using random noise.
3. Train the Discriminator on both real and fake images.
4. Train the Generator to fool the Discriminator.
5. Repeat the adversarial training for multiple epochs.
6. Save generated image samples every five epochs.
7. Save the trained Generator and Discriminator models.

## Results

The Generator gradually improved its ability to generate handwritten digit images through adversarial training. Generated samples were saved after every five epochs to observe the learning progress. Although the initial outputs appeared as random noise, the Generator learned increasingly meaningful digit patterns over time.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## Conclusion

This project demonstrates the implementation of a basic Generative Adversarial Network for handwritten digit generation. It highlights the adversarial learning process between the Generator and Discriminator and provides practical experience with generative deep learning models.

