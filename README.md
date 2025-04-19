# da6401_assignment2

wandb report link: https://wandb.ai/da24m004-iitmaana/inaturalist-cnn-sweep_01/reports/DA6401-Assignment-2--VmlldzoxMjM1MTU4Mg?accessToken=1dqoxs39cnvul3uf5u67wu2y8z75zn4vdoo07m5wuwnn90z8795pw7323cb5vxza

# PART A

## Features

- **Automated Dataset Handling:** Downloads and extracts the iNaturalist-12K dataset if not present.
- **Flexible CNN Architecture:** Easily configure the number of filters, filter sizes, activation functions, dense layer size, and filter organization strategy.
- **Data Augmentation:** Optional, with various transformations to improve generalization.
- **Hyperparameter Sweeps:** Automated tuning using wandb sweeps.
- **Experiment Tracking:** Logs training metrics, configurations, and artifacts to wandb.
- **Visualization:** Plots model predictions for each class.


## Dataset

- **Source:** [iNaturalist-12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)
- **Structure:** Standard ImageFolder format with `train` and `val` directories.


## Installation

Install the required packages:`pip install torch torchvision wandb tqdm scikit-learn matplotlib`

## Usage

### 1. Dataset Download and Preparation

The script checks for the dataset locally. If not found, it downloads and extracts it automatically:
```python
dataset_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
dataset_zip_path = "/kaggle/working/nature_12K.zip"
dataset_dir = "nature_12K"
```

### 2. Data Loading and Preprocessing

Images are resized, normalized, and optionally augmented. The training set is split into training and validation subsets.
```python
def prepare_datasets(data_dir, val_split=0.2, batch_size=32, image_size=(224, 224)):
# Applies transforms, loads ImageFolder datasets, splits train/val, returns tensors
```

### 3. Flexible CNN Model

The core model is defined in the `FlexibleCNN` class, supporting:

- Configurable number and size of filters per layer.
- Custom activation functions: relu, leakyrelu, gelu, silu, mish.
- Batch normalization and dropout.
- Filter organization strategies: 'same', 'double', 'half', 'pyramid'.
- Dense layer with configurable size.
- Output layer with 10 neurons (for 10 classes).

**Model Structure:**
- 5 blocks of Conv2d → (BatchNorm) → Activation → MaxPool2d → Dropout
- Flatten
- Dense (fully connected) layer
- Output layer

```python
class FlexibleCNN(nn.Module):
def init(self, num_filters=32, filter_size=3, activation='relu', dense_neurons=512,
input_channels=3, num_classes=10, use_batch_norm=True, dropout_rate=0.2,
filter_strategy='same'):
# Build the model as per configuration
```


### 4. Training and Validation

The training loop logs loss and accuracy to wandb, and saves the best model based on validation accuracy.

```python
def train(config=None):
# Loads data, initializes model, optimizer, criterion
# Trains for config['epochs'], logs to wandb, saves best model
```

### 5. Hyperparameter Sweep

A sweep configuration is defined for wandb to optimize hyperparameters such as number of filters, filter size, activation, dense layer size, learning rate, batch size, batch normalization, dropout, data augmentation, and filter strategy.


### 6. Best Model Training and Evaluation

After the sweep, the best configuration is used to retrain the model for up to 100 epochs, and the model is evaluated on the test set.

```python
def train_best_model(sweep_id, entity, project, epochs=100):
# Loads best sweep config, trains model, evaluates on test set, logs results
```


### 7. Visualization

A grid of images is plotted for each class, showing true and predicted labels, with predictions colored green (correct) or red (incorrect).

```python
def plot_class_predictions(model, test_loader, test_dataset, num_classes=10, samples_per_class=3):
# Plots a grid of images with predicted and true class labels
```
 

---

## Code Structure

| Section                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Dataset Preparation      | Download, extract, and preprocess images                                    |
| Model Definition         | FlexibleCNN class with customizable architecture                            |
| Training Loop            | Training, validation, and metric logging                                    |
| Hyperparameter Sweep     | wandb sweep setup for automated tuning                                      |
| Best Model Evaluation    | Retrain and evaluate best configuration                                     |
| Visualization            | Plotting class predictions for inspection                                   |

---

## Notes

- The code is suitable only for Kaggle environments.
- Update dataset paths if running locally.
- Requires a valid wandb API key for experiment tracking.

## How the Code Works 

1. **Dataset Handling:**  
   Checks if the iNaturalist-12K dataset exists locally. If not, it downloads and extracts the dataset automatically.

2. **Data Preprocessing:**  
   Images are resized to 224x224, normalized, and optionally augmented with random flips, rotations, and color jittering.

3. **Model Construction:**  
   The `FlexibleCNN` class builds a 5-layer CNN with configurable filter counts, filter sizes, activation functions, batch normalization, dropout, and a dense layer. The filter organization strategy allows for increasing, decreasing, or constant filter counts across layers.

4. **Training Loop:**  
   For each epoch, the model is trained on the training set and evaluated on the validation set. Training and validation metrics are logged to wandb. The model with the highest validation accuracy is saved.

5. **Hyperparameter Sweep:**  
   wandb's Bayesian optimization is used to find the best hyperparameters. The sweep runs multiple experiments, each with a different configuration.

6. **Best Model Evaluation:**  
   The best configuration from the sweep is used to retrain the model for more epochs. The final model is evaluated on the test set, and test accuracy is logged.

7. **Visualization:**  
   A function plots a grid of test images for each class, showing both the true and predicted labels, making it easy to visually assess model performance.


# PART B

## Overview

The code creates an image classification pipeline for a 10-class subset of the iNaturalist-12K dataset. It uses transfer learning by fine-tuning only the final layer of a pre-trained ResNet50 model, ensuring efficient training and strong performance.

## Dataset Download & Preparation

- **Source:** The iNaturalist-12K dataset is downloaded from a specified URL if not already present.
- **Extraction:** The dataset is extracted from a ZIP file into the working directory.
- **Structure:** The dataset is expected to have `train` and `val` folders, each containing subfolders for each class.

**Automated Steps:**
- Checks if the dataset exists locally.
- Downloads and extracts the dataset if missing, using `requests` and `zipfile`.
- Progress is shown with `tqdm`.

## Data Loading and Preprocessing

- **Transforms:** Images are resized to 224x224, converted to tensors, and normalized with mean and std of 0.5 for each RGB channel.
- **Splitting:** The training data is split into training and validation sets (default 80%/20%).
- **DataLoaders:** PyTorch DataLoaders are used for efficient batching and shuffling.
```python
def prepare_datasets(data_dir, val_split=0.2, batch_size=32, image_size=(224, 224)):
# Applies transforms, splits data, and returns train, val, and test DataLoaders
```

## Model Architecture & Modification

- **Base Model:** Uses a pre-trained ResNet50 from torchvision.
- **Transfer Learning:** All layers are frozen except the final fully connected layer.
- **Output Layer:** The final layer is replaced with a new `nn.Linear` layer for 10 output classes.

**Key Steps:**
- Load ResNet50 with pretrained weights.
- Replace the final classification layer.
- Freeze all other parameters.

## Training Procedure

- **Loss Function:** Cross-entropy loss for multi-class classification.
- **Optimizer:** Adam optimizer, updating only the new final layer.
- **Device:** Uses GPU if available, otherwise CPU.
- **Epochs:** Default is 10 epochs (can be changed).
- **Training Loop:** Alternates between training and validation each epoch, tracking and printing loss and accuracy.
- **Best Model:** Saves the model weights with the highest validation accuracy.

```python
def train_model(model, criterion, optimizer, num_epochs=40):
# Trains and validates the model, returns the best model
```

## Usage Instructions

1. **Dependencies:**
   - PyTorch
   - torchvision
   - numpy
   - pandas
   - tqdm
   - requests
   - scikit-learn

2. **Run the Script:**
   - Place the script in your working directory and run with Python 3.
   - The script will automatically download and extract the dataset if necessary.

3. **Adjust Parameters:**
   - Change `num_classes`, `batch_size`, `num_epochs`, `learning_rate` at the top of the script as needed.

## Key Parameters

| Parameter      | Description                               | Default Value    |
|----------------|-------------------------------------------|------------------|
| num_classes    | Number of output classes                  | 10               |
| batch_size     | Batch size for DataLoader                 | 16 or 64         |
| num_epochs     | Number of training epochs                 | 10               |
| learning_rate  | Learning rate for optimizer               | 0.001            |
| image_size     | Image resize dimensions                   | (224, 224)       |
| val_split      | Fraction of training set for validation   | 0.2              |


## Notes

- **Transfer Learning:** Only the final layer is trained, making training fast and requiring less data.
- **Dataset Path:** The script expects the dataset to be extracted to `/kaggle/working/inaturalist_12K` by default.
- **Customization:** You can adjust the number of classes, batch size, and other parameters as needed.
- **Logging:** Basic print statements are used for progress reporting; for advanced tracking, integrate tools like Weights & Biases.


## Script Workflow Summary

1. **Setup:** Import libraries and define parameters.
2. **Dataset Handling:** Download and extract if needed.
3. **Data Preparation:** Apply transforms, split into train/val/test, create DataLoaders.
4. **Model Setup:** Load pre-trained ResNet50, freeze all except the last layer, adapt to 10 classes.
5. **Training:** Train only the final layer, validate after each epoch, save the best model.
6. **Output:** The best model (based on validation accuracy) is ready for inference or further fine-tuning.

