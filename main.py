import numpy as np
from tqdm import tqdm
from python_speech_features import mfcc
import os
import operator
import pickle
import scipy.io.wavfile as wav
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split


def extract_features(path_to_dataset):
    i = 0
    with open("genres_data.dmp", "wb") as handle:
        for folder in os.listdir(path_to_dataset):
            i += 1
            if i == 11:
                break
            for file in os.listdir(path_to_dataset + "/" + folder):
                # fetching sampling rate and data from each audio file
                (rate, sig) = wav.read(path_to_dataset + "/" + folder + "/" + file)
                # analyzing files divided into 20 ms parts
                mfcc_features = mfcc(sig, rate, winlen=0.02, winstep=0.01, numcep=15, nfft=1200, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_features))
                mean_matrix = np.round(mfcc_features.mean(axis=0), 3)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, handle)
    handle.close()


def check_distance(point1, point2, k):
    dist = np.trace(np.linalg.inv(point2[1]) @ point1[1])
    dist += (((point2[0] - point1[0]).transpose() @ np.linalg.inv(point2[1])) @ (point2[0] - point1[0]))
    dist += np.log(np.linalg.det(point2[1])) - np.log(np.linalg.det(point1[1]))
    dist -= k
    return dist


def get_neighbours(train_data, considered_point, k):
    distances = []
    for x in train_data:
        dist = check_distance(x, considered_point, k) + check_distance(considered_point, x, k)
        distances.append((x[2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for neighbour in range(k):
        neighbours.append(distances[neighbour][0])
    return neighbours


def get_nearest_neighbour(neighbours):
    classes = {}
    for neighbour in neighbours:
        if neighbour in classes:
            classes[neighbour] += 1
        else:
            classes[neighbour] = 1
    sorted_classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classes[0][0]


def load_dataset(filename):
    dataset = []
    with open(filename, 'rb') as file:
        try:
            while True:
                dataset.append(pickle.load(file))
        except EOFError:
            pass
    return dataset


def predict_sample_data(file, genres_names):
    dataset = load_dataset("./genres_data.dmp")
    train_dataset, test_dataset = train_test_split(dataset, test_size=1, random_state=80)
    (rate, sig) = wav.read(file)
    # length = sig.shape[0] / rate
    # print(f"length = {length}s")
    mfcc_features = mfcc(sig, rate, winlen=0.02, winstep=0.01, numcep=15, nfft=1200, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_features))
    mean_matrix = np.round(mfcc_features.mean(axis=0), 3)
    feature = (mean_matrix, covariance, 0)
    pred = get_nearest_neighbour(get_neighbours(train_dataset, feature, 5))
    return genres_names[pred-1]


def main():
    path_to_dataset = './dataset'
    genres_names = []

    # extract features to .dmp file
    if not os.path.isfile('genres_data.dmp'):
        extract_features(path_to_dataset)

    # get folders names
    for folder in os.listdir("./dataset/"):
        genres_names.append(folder)

    # test dataset files & check accuracy
    dataset = load_dataset("./genres_data.dmp")
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=80)
    predictions = []
    test_pred = []
    length = len(test_data)
    for x in tqdm(range(length)):
        neighbour = get_nearest_neighbour(get_neighbours(train_data, test_data[x], 5))
        test_pred.append(genres_names[test_data[x][-1] - 1])
        predictions.append(genres_names[neighbour - 1])
    accuracy_count = accuracy_score(test_pred, predictions, normalize=False)
    print("%s: %f" % ('F1 score is', f1_score(test_pred, predictions, average='weighted')))
    print("%s: %f" % ('Precision score is', precision_score(test_pred, predictions, average='weighted', zero_division=1)))
    print("Accuracy: %0.2f%c" % (100 * accuracy_count / length, '%'))

    # test samples
    print('')
    path_to_sample = './test_samples/country.wav'

    sample_genre = predict_sample_data(path_to_sample, genres_names)
    print("%s: %s" % ('The genre of your sample is', sample_genre))


if __name__ == "__main__":
    main()

