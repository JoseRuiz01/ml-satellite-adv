# Adversarial Attacks on ML-based Satellite Image Classification

## Project Overview

This project builds a baseline **image classification model** using satellite imagery from the **EuroSAT dataset** and explores adversarial attacks on these models. To study robustness, different adversarial attack methods are implemented against the classifier, followed by the development of defensive strategies designed to mitigate these attacks and maintain model performance.

---

## Dataset

The project uses the **EuroSAT dataset**, which contains high-resolution RGB satellite images organized into class folders (e.g., Forest, Highway, Industrial). The dataset was downloaded and stored in `data/raw`.

The images are `.tif` format, and the dataset contains 10 different classes.

---

## Classification Model

A baseline **ResNet18** model is used:

* Initialized with pretrained ImageNet weights.
* Fully connected layer adapted to 10 classes.
* Trained using **cross-entropy loss** and **Adam optimizer**.
* Default training:
    * *Epochs*: configurable (default 10).
    * *Batch size*: configurable (default 64).
    * *Learning rate*: configurable (default 1e-4).

