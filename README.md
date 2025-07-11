1. Introduction
In this project, we were given the task of designing a program that could automatically
classify the digits in CAPTCHA images. The CAPTCHA images
contain three handwritten digits, often distorted with noise, overlapping strokes,
and cluttered backgrounds. Our goal was to extract features from these images
and classify the digits using traditional image processing and machine learning
techniques, without relying on ocr or deep neural networks.
To begin the project, we visually inspected several CAPTCHA images from the
provided training set. We noticed that the images had artifacts such as curved lines
crossing over the digits, variable contrast, and sometimes merging digits, which
made segmentation difficult. Furthermore, digits were occasionally rotated, scaled,
or slightly deformed, adding further complexity to the classification task.
We were provided with 1200 training images and corresponding labels, and our goal
was to build a pipeline that extracts features for each digit and predicts the digits
using a classifier. The final evaluation was done after our implementation. From
our initial experiments and inspection, the most main challenges we identified were
digit segmentation (especially when digits are touching or overlapping) and robust
feature extraction under varying transformations and noise.
2. Method
To begin processing the images, we first designed a preprocessing step to reduce
noise and highlight the digits more clearly. One interesting part of this was applying
a filtering technique in the frequency domain using the Fourier Transform.
What this means is that we converted the image from its original pixel-based representation
into a form that shows the frequency content of the image. By doing
this, we could identify and suppress specific frequency bands that usually carried
unwanted noise or mess. We applied a bandpass filter that removed very low frequencies,
which often correspond to large-scale background gradients, as well as
high frequencies that tend to contain sharp noise. The filtered image was then
converted back into the spatial domain, where the digit structures appeared more
distinct and less noisy. This step improved the results of our later binarization and
segmentation steps, as it helped remove distractions that would otherwise merge
with the digits.
After this frequency-based cleanup, we converted the image to grayscale and applied
a median filter to further reduce any remaining noise. We then binarized the
image using Otsu’s thresholding method and inverted the result so that the digits
became white objects on a black background. This was followed by morphological
operations, including erosion and dilation, which helped remove small noise blobs
and enhance the connectedness of the digit regions.
The next step involved segmenting the three digits in each image. Since CAPTCHA
images are visually unpredictable, we couldn’t rely on fixed-size boxes or templates.
Instead, we used connected component analysis to detect regions in the image that
appeared to be digits. When only one connected component was detected, which
usually meant all three digits were touching, we split the region into three equal
vertical parts. If two components were found, we split the larger one into two parts
and used the other as a separate digit. If three or more components were detected,
we took the first three and treated each as a digit. This segmentation logic was
designed to be flexible and worked well across most images in the dataset.
Once we had extracted the three digit regions from an image, we needed to describe
each digit using features that could help a machine learning model tell them apart.
For each digit, we extracted a combination of shape-based and moment-based
features. The shape-based features included properties like the area of the digit,
how circular it was, its solidity, eccentricity, and the position of its center. These
features describe the general shape and layout of the digit. In addition to these, we
calculated a set of values called Hu moments. These are mathematical quantities
that summarize the shape of a binary image in a way that is invariant to rotation,
scaling, and translation. That means the same digit would give roughly the same
Hu moments even if it appeared at a different angle or size. We extracted nine Hu
moments and combined them with six shape features to create a 15-dimensional
feature vector for each digit. With three digits per image, this resulted in a 45-
dimensional feature vector for the full image.
3. Implementation
All the steps mentioned above were implemented as a modular pipeline in MATLAB.
The preprocessing, segmentation, and feature extraction were handled inside
the FeatureExtraction.m script. This script takes a CAPTCHA image and returns
a 3 × 15 matrix containing features for each digit. This matrix is flattened
into a single 1 × 45 feature vector.
The script prepare_data.m was used to iterate over the training dataset, apply
the feature extraction process, and save all features and corresponding labels
into a .mat file. For training, we used train_model.m, which loaded the features
and labels, reshaped them, and trained a multi-class AdaBoost classifier.
The weak learners were decision trees, and we set the maximum number of splits
(MaxNumSplits) to 10 to reduce the risk of overfitting. The model was saved as
AdaBoostModelv2.mat.
During testing, the evaluate_classifier.m script loaded each test image, extracted
its features using the same pipeline, and applied the trained model from
myclassifier.m to predict all three digits at once. The script compared predictions
with ground-truth labels and printed the overall accuracy.
The Fourier filtering was implemented through helper functions like fourier_tool.m
and spec_filter_image.m, which handled FFT transformation, masking of specific
frequency bands, and conversion back to spatial domain. These functions
played an important role in cleaning up background interference before segmentation
