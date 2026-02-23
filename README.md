#  Learning Probability Density Functions using GAN

##  Project Overview

This project learns an unknown probability density function (PDF) using only data samples — without assuming any parametric distribution (Gaussian, Exponential, etc.).

We use:
- Real-world NO₂ concentration data
- A nonlinear transformation
- A Generative Adversarial Network (GAN)
- Kernel Density Estimation (KDE) for PDF approximation

The GAN implicitly learns the distribution of the transformed variable.

##  Objective

To learn the unknown probability density function of a transformed variable:

    z = x + a_r * sin(b_r * x)

Where:

- x = NO₂ concentration  
- a_r = 0.5 * (r mod 7)  
- b_r = 0.3 * ((r mod 5) + 1)  
- r = University Roll Number  

The GAN must:
- Learn distribution only from samples
- Assume no analytical form
- Generate samples matching the real distribution

##  Dataset

Dataset: India Air Quality Data

Feature used:
- NO2 concentration


##  Methodology

### Step 1 — Data Transformation

Each NO₂ value is transformed as:

    z = x + a_r * sin(b_r * x)

This creates a nonlinear distribution.


### Step 2 — GAN Architecture

Generator:
- Input: 5-dimensional Gaussian noise
- Dense(5 → 32) + ReLU
- Dense(32 → 64) + ReLU
- Dense(64 → 1)

Discriminator:
- Dense(1 → 64) + LeakyReLU
- Dense(64 → 32) + LeakyReLU
- Dense(32 → 1) + Sigmoid

Loss Function:
- Binary Cross Entropy

Optimizer:
- Adam (learning rate = 0.0002)

---

### Step 3 — PDF Approximation

After training:
1. Generate 10,000 samples from the generator
2. Estimate density using Kernel Density Estimation (KDE)
3. Compare GAN PDF vs real histogram

---

##  Results

The final output includes:
- Histogram of real transformed data
- KDE curve from GAN-generated samples
- Visual comparison of distribution match

<img width="872" height="477" alt="Screenshot 2026-02-23 at 1 23 30 PM" src="https://github.com/user-attachments/assets/a1e0939b-aac9-456f-9300-ea2783cabd8d" />
