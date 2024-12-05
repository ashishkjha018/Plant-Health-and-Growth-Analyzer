# Plant-Health-and-Growth-Analyzer
Developed a machine learning-based application in Python that analyzes plant leaf images to assess their health and identify potential diseases. The system leverages advanced data models to provide accurate diagnostics, supporting better plant care and early disease detection.

## Overview

This project focuses on the detection of diseases in various types of plants, including Apple, Cherry, Grape, Strawberry, and Potato. The goal is to identify and categorize diseases in these plants to facilitate timely intervention and prevent further damage. The diseases and respective categories for each plant are as follows:

- **Apple:** ["scab", "rot", "Healthy"]
- **Cherry:** ["mildew", "Healthy"]
- **Grape:** ["Measels", "Leaf Blight", "Healthy"]
- **Strawberry:** ["Leaf Scorch", "Healthy"]
- **Potato:** ["Early Blight", "Late Blight", "Healthy"]

The images used for training and testing are of size 224x224 pixels.

## Model Used

In this project, the VGG16 (Visual Geometry Group 16) model has been employed for plant disease detection. VGG16 is a widely-used convolutional neural network architecture known for its simplicity and effectiveness in image classification tasks.

## Dataset

The dataset has been collected from Kaggle and comprises images of plants with corresponding labels indicating the presence of diseases or a healthy state. The dataset has been divided into training and testing sets to evaluate the model's performance.

## Usage

To use the model for plant disease detection, follow these steps:

1. Clone the repository: 
