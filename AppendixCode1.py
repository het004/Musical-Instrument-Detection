import fnmatch
import numpy as np
import itertools
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix, classification_report
import threading
import joblib

np.random.seed(1)

data_path = './dataset'

def get_audio_files(path, extension='*.mp3'):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, extension):
            files.append(os.path.join(root, filename))
    return files

audio_files = get_audio_files(data_path)

classes = ['flute', 'sax', 'oboe', 'cello', 'trumpet', 'viola']
color_dict = {'cello': 'blue', 'flute': 'red', 'oboe': 'green', 'trumpet': 'black', 'sax': 'magenta', 'viola': 'yellow'}

def get_labels_and_colors(files, classes, color_dict):
    labels = []
    colors = []
    for file in files:
        for cls in classes:
            if cls in file:
                labels.append(cls)
                colors.append(color_dict[cls])
                break
        else:
            labels.append('other')
            colors.append('gray')
    return labels, colors

labels, colors = get_labels_and_colors(audio_files, classes, color_dict)

label_encoder = LabelEncoder()
classes_num = label_encoder.fit_transform(labels)

fs = 44100
n_mels = 128
n_mfcc = 13

def extract_features(file, fs=44100, n_mels=128, n_mfcc=13):
    y, sr = librosa.load(file, sr=fs)
    y /= np.max(np.abs(y))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

feature_vectors = [extract_features(file) for file in audio_files]

scaler = StandardScaler()
scaled_feature_vectors = scaler.fit_transform(feature_vectors)

X_train, X_test, y_train, y_test = train_test_split(scaled_feature_vectors, classes_num, test_size=0.25, random_state=0, stratify=classes_num)

k_neighbors = 1
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)

def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)

thread = threading.Thread(target=train_classifier, args=(knn_classifier, X_train, y_train))
thread.start()
thread.join()

model_dict = {'model': knn_classifier, 'scaler': scaler, 'label_encoder': label_encoder}
joblib.dump(model_dict, 'music.joblib')

predicted_labels = knn_classifier.predict(X_test)
print(predicted_labels)

recall = recall_score(y_test, predicted_labels, average=None)
precision = precision_score(y_test, predicted_labels, average=None)
f1 = f1_score(y_test, predicted_labels, average=None)
accuracy = accuracy_score(y_test, predicted_labels)

print("Recall:", recall)
print("Precision:", precision)
print("F1-Score:", f1)
print("Accuracy:", accuracy)

class_names = label_encoder.classes_
print(classification_report(y_test, predicted_labels, target_names=class_names))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, predicted_labels)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

plt.show()
