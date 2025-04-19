# da6401_assignment2

# PART A

## Overview

The notebook automates hyperparameter tuning for convolutional neural networks (CNNs) on a classification task. It uses wandb sweeps to systematically explore different model and training configurations, logging each run for analysis and comparison.

## 1. Library Imports and Environment Setup

- **Deep Learning Frameworks:** TensorFlow and Keras are imported for model building and training.
- **Data Handling:** Libraries like NumPy and Pandas are used for data manipulation.
- **Experiment Tracking:** wandb is initialized for logging metrics, configurations, and managing the sweep process.

## 2. Hyperparameter Sweep Configuration

- A configuration dictionary defines the sweep space, specifying:
  - **Activation Functions:** `'relu'`, `'silu'`, `'gelu'`, `'mish'`
  - **Batch Sizes:** e.g., 32, 64, 128
  - **Dense Layer Neurons:** e.g., 128, 256
  - **Dropout Rates:** e.g., 0.2, 0.3, 0.4
  - **Epochs:** fixed at 7
  - **Convolution Filter Size:** typically 3
  - **Filter Strategies:** `'same'`, `'half'`, `'pyramid'`
  - **Learning Rate:** sampled from a continuous range
  - **Number of Filters:** e.g., 16, 32, 64
  - **Data Augmentation and Batch Normalization:** boolean flags
  - **Random Seed:** for reproducibility

## 3. Data Preparation

- **Loading and Splitting:** The dataset is loaded and split into training and validation sets.
- **Preprocessing:** Input features are normalized or standardized.
- **Augmentation:** If enabled, data augmentation is applied to the training set.

## 4. Model Construction

- A function builds the CNN model according to the current hyperparameter configuration.
  - **Convolutional Layers:** Number, size, and strategy are set based on sweep parameters.
  - **Activation & Dropout:** Layer activations and dropout rates are configurable.
  - **Batch Normalization:** Included if enabled in the configuration.
  - **Dense Layer:** Size is set by the sweep.

## 5. Training and Logging

- **Compilation:** The model is compiled with the optimizer and learning rate from the sweep config.
- **Training Loop:** The model trains for the specified epochs, with metrics logged to wandb.
- **Logging:** Training/validation accuracy and loss are tracked for each run.

## 6. Sweep Agent Execution

- The wandb sweep agent:
  - Samples hyperparameter combinations.
  - Runs the training function for each configuration.
  - Logs all metrics and artifacts for
 

# PART B

