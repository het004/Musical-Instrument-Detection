# Musical-Instrument-Detection
A Python-based project using MFCC features and KNN classifier to detect musical instruments from audio files.


# Musical Instrument Detection

This project implements a system for identifying musical instruments from audio files using machine learning. The system extracts Mel-frequency cepstral coefficients (MFCC) from audio signals and trains a K-Nearest Neighbors (KNN) classifier to classify different musical instruments.

## Features:
- Musical instrument detection using audio data.
- Uses MFCC for feature extraction.
- Implements a KNN classifier for classification.
- Trained on a dataset of musical instrument audio files.
- Saves the trained model for inference on new audio files.

## Dependencies:
- Python 3.x
- Numpy
- Scikit-learn
- Librosa
- Matplotlib
- Seaborn
- Joblib

## Running the Project:
1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Prepare the dataset (audio files for instruments) and place it in the `dataset/` folder.
4. Run the training script using `python train.py`.
5. Use the saved model to classify new audio files by running `python inference.py`.

## Code Structure:
- `train.py`: Main script for training the KNN classifier.
- `inference.py`: Script to make predictions on new audio files using the saved model.
- `music.joblib`: Saved model file for inference.
- `requirements.txt`: List of required Python libraries.

## Results:
The classifier achieved an accuracy of XX% on the test set.

## Future Improvements:
- Exploring additional features beyond MFCC for better classification.
- Using more advanced machine learning models to handle overlapping frequencies between instruments.
