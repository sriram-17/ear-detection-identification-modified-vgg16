# A Deep Learning Model for Ear Detection and Identification Using Modified VGG-16

## Abstract

In the modern digital age, security remains a paramount concern. Biometrics have become a popular solution due to their numerous benefits across various sectors. Ear biometrics, where the ear is used as a unique identifying feature, is particularly advantageous as the ear is often more visible than the entire face in many scenarios. We introduce a Deep Neural Network (DNN) model for Ear Detection and Identification using a modified VGG-16 architecture. The model begins with preprocessing ear images from the dataset, followed by training a hybrid model that integrates the modified VGG-16 with some early layers frozen. The system is designed to identify individuals based on preprocessed images, demonstrating improved recognition specificity.

**Keywords**: Normalization, Biometrics, DNN, VGG-16.

## Objectives

- **Ear Detection**: Identifying the presence of ears in images.
- **Ear Recognition**: Identifying specific individuals based on ear images.

## Scope

- **Image Resolution**: Images should be at least 224 x 224 pixels.
- **Limitations**: The system may struggle with images obscured by hair or ear piercings.

## Basic Concepts

### Image Processing and Computer Vision

Computer Vision involves understanding and manipulating images and videos, providing the foundation for many artificial intelligence applications.

### PyTorch

PyTorch is an open-source machine learning framework based on the Torch library, utilized for applications in computer vision and natural language processing.

### Deep Learning

Deep learning models address challenges in feature extraction by mimicking human brain functions to learn from experience.

## Dataset Description

The dataset is derived from the AMI dataset ([AMI Ear Database](https://ctim.ulpgc.es/research_works/ami_ear_database/#whole)) and includes images of ears cropped from an unconstrained environment. It is organized into the following folders:
- **Train**: 12 images per person, total of 1200 images.
- **Test**: 2 images per person, total of 200 images.
- **Valid**: 4 images per person, total of 400 images.

## Drawbacks

- **Vanishing Gradient Problem**: Can occur in deep neural networks.
- **Higher Loss**: May result from the complexity of the model.
- **Slow Learning Rate**: Due to the large number of weights in the network.

## Implementation

### Module 1: Data Preprocessing and Model Generation

**Transformations for Training Data**:
- **RandomResizedCrop**: Crops random areas of size 256x256 pixels and scales them.
- **RandomRotation**: Rotates images by up to 15 degrees.
- **CenterCrop**: Crops the central part of the image to 224x224 pixels.
- **ToTensor**: Converts images to tensors.
- **Normalize**: Normalizes images to reduce skewness and improve learning efficiency.

**Transformations for Test and Validation Data**:
- **Resize**: Crops to 256x256 pixels.
- **CenterCrop**: Crops to 224x224 pixels.
- **ToTensor**: Converts images to tensors.
- **Normalize**: Normalizes images as described above.

**Data Processing**:
- **Class Mapping**: Uses `class_to_idx` to map persons to ear indices.
- **Loss Function**: Negative Log-Likelihood Loss (NLLLoss).
- **Optimizer**: Adam Optimizer.
- **Model Training**: Hybrid model is trained with the above transformations, and the model is saved for further use.

### Module 2: Identification of Individual and Calculation of Probability

**Image Processing**:
- Convert test images to the required size and resolution.
- Pass the processed image through the model to identify the class.
- Calculate the probability for each class and select the class with the highest probability.

**Architecture – Modified VGG16**:
- **Convolutional Layers**:
  - 2 x Conv Layer (64 channels, 3x3 kernel)
  - 1 x Max Pool Layer (2x2 pool size)
  - 2 x Conv Layer (128 channels, 3x3 kernel)
  - 1 x Max Pool Layer (2x2 pool size)
  - 3 x Conv Layer (256 channels, 3x3 kernel)
  - 1 x Max Pool Layer (2x2 pool size)
  - 2 x Conv Layer (512 channels, 3x3 kernel)
  - 1 x Max Pool Layer (2x2 pool size)
- **Fully Connected Layers**:
  - 4 x `nn.Linear` Layers
  - 4 x Dropout Layers (0.4 probability)
  - 1 x LogSoftmax Layer (dimension 1)

**Fine-Tuning**:
- Fine-tuning adjusts model parameters for better accuracy.
- Utilizes an early stopping method to prevent overfitting and improve prediction rates.
- Fine-tuned model shows improved accuracy compared to the initial model.

## Test Cases

- **Images**: (Include sample images used for testing)

## Results

- **Accuracy**: 94%
- **Training and Validation Loss**: 0.05
- **Precision**: 73.94%
- **Recall**: 63.5%
- **F1 Score**: 63.64%

## Contact

For inquiries or feedback, please reach out to us at:
- **Email**: sriram.pulivarthi.3@gmail.com

