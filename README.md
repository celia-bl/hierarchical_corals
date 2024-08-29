# Comparison of Hierarchical and Flat Classifiers

## Overview
This repository contains the implementation and results of a comparative study between hierarchical and flat classifiers using a new dataset of Brazilian benthic images. The aim is to evaluate the performance of these classifiers and determine which approach yields better results for the given problem.

## Dataset
The dataset comprises 1,549 images collected and annotated by the Marine Ecology Laboratory/UFRN at Rio do Fogo, Northeast Brazil. Between 2017 and 2022, 100 images were collected from each of two sites every three months. Each image contains 25 random annotated points, resulting in 38,725 annotations across 54 distinct labels, which are highly imbalanced. Specifically, 11 labels account for 95% of the annotations, while 24 classes have fewer than 20 annotations each. The images can be downloaded from the provided link. All experiments for analyzing these images are available in `analyze_data.py`.

## Feature Extraction
To extract features, a 224x224 image patch is cropped around each annotated point. This process is handled in `img_transformation.py`. These patches are then processed through an EfficientNet B0 model, with weights customized by CoralNet, trained on 16 million benthic images. The resulting feature vectors are used for training the classifiers.

## Flat Classifier
Our baseline is a standard Multi-Layer Perceptron (MLP) classifier provided by the scikit-learn library ([scikit-learn](https://github.com/scikit-learn/scikit-learn.git)). The MLP has two neural network layers with 200 and 100 units, respectively.

## Hierarchical Classifier
The hierarchical classifier is implemented using `hiclass` ([hiclass](https://github.com/scikit-learn-contrib/hiclass.git)), which also depends on scikit-learn classifiers. We opted for a Local Classifier per Parent Node, meaning a separate classifier for each node with children. Each of these classifiers is an MLP identical to our baseline, with two layers of 200 and 100 units.

## Metrics
The metrics used to compare the two approaches are the F1-score and a hierarchical score. The F1-score is computed using scikit-learn, while the hierarchical score is computed using `hiclass`. The hierarchical score penalizes errors less when elements are close in the hierarchical tree.

## Setup
To run the tests, first extract the features with:

```python
from extract_features import * 

annotation_file = '/path/to/annotation/file'
image_file = '/path/to/images'
weight_file = '/path/to/weight/file'

batch = create_full_batch(annotation_file, image_file) 
extract_features(batch, crop_size=224, input_folder=image_file, model_name='efficientnet-b0', weights_file=weight_file)
```
Once the features are extracted, you can run the main script, optionally adjusting the number of images to test.
