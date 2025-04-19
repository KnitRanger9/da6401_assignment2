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

## 1. Environment Setup and Imports

- The code begins by importing essential libraries:
  - **NumPy** and **Pandas** for data handling.
  - **os** and **pathlib** for file and directory operations.
  - **torch**, **torchvision**, and related modules for deep learning.
  - **requests** and **zipfile** for downloading and extracting datasets.
  - **tqdm** for progress bars.
  - **wandb** for experiment tracking (though not actively used in the script).
  - **time** and **copy** for training utilities.

## 2. Dataset Download and Extraction

- **Dataset URL and Paths:**  
  The dataset (iNaturalist-12K) is downloaded from a public URL if not already present in the working directory.
- **Download Logic:**  
  Uses `requests.get` with streaming and `tqdm` to show download progress. The file is saved as a zip archive.
- **Extraction:**  
  If the dataset directory does not exist, the zip file is extracted into the current directory using `zipfile.ZipFile`.

## 3. Data Preparation and Loading

- **Transformations:**  
  Images are resized to 224x224, converted to tensors, and normalized to have mean and std of 0.5 for each channel.
- **Dataset Structure:**  
  Uses `ImageFolder` to load images from the extracted directories (`train` and `val`).
- **Train/Validation Split:**  
  The training set is split into training and validation subsets using `random_split` with an 80/20 split.
- **DataLoaders:**  
  PyTorch DataLoaders are created for train, validation, and test sets, with batch size 64 and pinning memory for efficiency.

## 4. Model Setup: Transfer Learning with ResNet50

- **Loading Pretrained Model:**  
  Imports ResNet50 from torchvision with pretrained weights.
- **Final Layer Replacement:**  
  The final fully connected (fc) layer is replaced with a new `nn.Linear` layer to output 10 classes, matching the dataset.
- **Freezing Layers:**  
  All layers except the new final layer are frozen (`requires_grad = False`), so only the last layer is updated during training.
- **Device Assignment:**  
  The model is moved to GPU if available, otherwise CPU.

## 5. Loss Function and Optimizer

- **Loss:**  
  Uses `nn.CrossEntropyLoss` for multi-class classification.
- **Optimizer:**  
  Adam optimizer is applied only to the parameters of the final layer (`model.fc.parameters()`), ensuring only the new layer is trained.

## 6. Training Loop

- **Function Definition:**  
  The `train_model` function encapsulates the training and validation process.
- **Epoch Loop:**  
  For each epoch, the model alternates between training and validation phases.
- **Batch Processing:**  
  For each batch, inputs and labels are moved to the appropriate device. The optimizer is zeroed, and a forward pass is performed.
- **Backward Pass:**  
  During training, gradients are computed and the optimizer steps. During validation, gradients are not tracked.
- **Metrics Calculation:**  
  Running loss and accuracy are accumulated for each phase.
- **Best Model Tracking:**  
  The model with the best validation accuracy is saved using a deep copy of its state dictionary.
- **Progress Reporting:**  
  Loss and accuracy for each phase are printed after every epoch.

## 7. Training Execution

- The training process is initiated by calling `train_model` with the model, loss function, optimizer, and number of epochs (10 by default).
- The best-performing model (on validation accuracy) is restored at the end of training.

## 8. Key Code Patterns

- **Transfer Learning Workflow:**


