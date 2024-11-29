# MNIST-CI Project

[![Build Status](https://github.com/[username]/MNIST-CI/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/[username]/MNIST-CI/actions/workflows/ml_pipeline.yml/)

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset, with a complete CI/CD pipeline using GitHub Actions.
The model is trained using several data augmentation techniques to improve robustness:

- Random rotation up to 20 degrees
- Gaussian blur with kernel size 5
- Normalization with mean=0.1307 and std=0.3081

These transformations help prevent overfitting by creating variations of the training images while preserving the essential digit features.


