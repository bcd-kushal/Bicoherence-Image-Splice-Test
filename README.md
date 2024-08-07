# Splice Image Detection using Bicoherence

> Model Accuracy: 72% - 83%

<br />

## Overview

This project implements a Python-based image splicing detection system using bi-coherence analysis and machine learning models. The goal is to identify tampered regions in images, which is useful for digital forensics and image verification tasks. The system uses bi-coherence to analyze the frequency domain of images and applies various machine learning classifiers to detect splicing.

## Features

- **Canny Edge Detection**: Utilizes the Canny edge detector to preprocess images for feature extraction.
- **Bi-Coherence Analysis**: Computes bi-coherence features in the frequency domain to identify anomalies.
- **Machine Learning Models**: Trains and evaluates different Support Vector Classifier (SVC) models to classify images as authentic or spliced.
- **Performance Evaluation**: Measures the accuracy of different SVC models on the test dataset.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/image-splicing-detection.git
   cd image-splicing-detection

2. **Create a Virtual Environment**

   ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. **Install Dependencies**

   ```txt
    numpy
    scipy  
    matplotlib
    scikit-learn
    opencv-python  
    pandas
    seaborn
    joblib

  Then install the dependencies

  ```python
  pip install -r requirements.txt
  ```

## Usage

### Data Preparation

Ensure you have the dataset available. The code expects two directories containing authentic and spliced images:

- `mini_authentic/:` Directory with authentic images
- `mini_spliced/:` Directory with spliced images

### Running the Detection
Mount Google Drive (if using Google Colab)

```py
from google.colab import drive
drive.mount('/content/drive')
Adjust the repo_path to point to your dataset directory on Google Drive.
```

### Run the Detection Script

Run the script to preprocess images, extract features, and evaluate the models:

```bash
Copy code
python detect_splicing.py
```

Ensure you have updated the path to your saved model in the joblib.load('') line.
