import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from sklearn import metrics
from util import other_class
from collections import Counter


from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


countMap = dict()
indexMap = dict()

class MNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)

# Mark data for deletion by marking as -1
def markForDeletion(data, startIndex, endIndex):
    for i in range(startIndex, endIndex):
        data[i] = -1
    return data


# Delete all data marked as -1
def deleteMarkedDataAndReturnNewData(data):
    data = data[data != -1]
    return data


# Balance Images
def balanceImages(images, deletedLabelIndexes):
    for index in deletedLabelIndexes:
        startIndex = (index - 1) * 28 * 28
        endIndex = index * 28 * 28
        images = markForDeletion(images, startIndex, endIndex)
    return deleteMarkedDataAndReturnNewData(images)


# Balance Labels
def balanceLabels(labelSet, minimumSampleCount):
    deletedIndexes = []
    for labels in countMap:
        if countMap[labels] > minimumSampleCount:
            while countMap[labels] != minimumSampleCount:
                indexToDelete = random.choice(indexMap[labels])
                startIndex = indexToDelete - 1
                endIndex = indexToDelete
                markForDeletion(labelSet, startIndex, endIndex)
                indexMap[labels].remove(indexToDelete)
                deletedIndexes.append(indexToDelete)
                countMap[labels] -= 1
    return deleteMarkedDataAndReturnNewData(labelSet), deletedIndexes


# Balance Images and Labels
def balanceDataset(images, labels, minimumSampleCount):
    labels, deletedIndexes = balanceLabels(labels, minimumSampleCount)
    images = balanceImages(images, deletedIndexes)
    return images, labels


# Add asymmetric noise
def addAsymmetricNoise(data, noise_ratio):
    source_class = [7, 2, 3, 5, 6]
    target_class = [1, 7, 8, 6, 5]
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(data == s)[0]
        n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        data[noisy_sample_index] = t
    return data

# Add symmetric noise
def addSymmetricNoise(data, noise_ratio):
    n_samples = data.shape[0]
    n_noisy = int(noise_ratio * n_samples / 100)
    class_index = [np.where(data == i)[0] for i in range(10)]
    class_noisy = int(n_noisy / 10)

    noisy_idx = []
    for d in range(10):
        noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
        noisy_idx.extend(noisy_class_index)

    for i in noisy_idx:
        data[i] = other_class(n_classes=10, current_class=data[i])
        
    return data



transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0), (1)),
    ])
transform_val = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0), (1))
    ])
   

train_dataset = datasets.MNIST(root='./data',download=True,train=True,transform=transform_train)
# Load from zip file -- Imbalanced data
training_images_unzipped = gzip.open('./data/MNIST/raw/train-images-idx3-ubyte.gz', 'r')
training_labels_unzipped = gzip.open('./data/MNIST/raw/train-labels-idx1-ubyte.gz', 'r')
test_images_unzipped = gzip.open('./data/MNIST/raw/t10k-images-idx3-ubyte.gz', 'r')
test_labels_unzipped = gzip.open('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz', 'r')

image_size = 28
training_labels_unzipped.read(8)
label_buf = training_labels_unzipped.read(60000)
labelArray = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)

training_images_unzipped.read(16)
buf = training_images_unzipped.read(image_size * image_size * 60000)
imbalanced_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)


test_labels_unzipped.read(8)
test_label_buf = test_labels_unzipped.read(10000)
test_labelArray = np.frombuffer(test_label_buf, dtype=np.uint8).astype(np.int32)

test_images_unzipped.read(16)
test_images_buf = test_images_unzipped.read(image_size * image_size * 10000)
test_images = np.frombuffer(test_images_buf, dtype=np.uint8).astype(np.float32)


for index, label in enumerate(labelArray):
    if label not in countMap:
        countMap[label] = 1
    else:
        countMap[label] += 1

    if label not in indexMap:
        indexMap[label] = list()
        indexMap[label].append(index)

    else:
        indexMap[label].append(index)

minimumSampleCount = countMap[min(countMap, key=countMap.get)]
#Balanced Without Noise
training_labels_balanced_without_noise = np.copy(labelArray)
balanced_data, training_labels_balanced_without_noise = balanceDataset(imbalanced_data, training_labels_balanced_without_noise, minimumSampleCount)

#Imbalanced Asymmetric
label_array_imbalanced_asym = np.copy(labelArray)
label_array_imbalanced_asym = addAsymmetricNoise(label_array_imbalanced_asym, 40)

#Balanced Symmetric
label_array_balanced_asym = np.copy(training_labels_balanced_without_noise)
label_array_balanced_asym = addAsymmetricNoise(label_array_balanced_asym, 40)

#Imbalanced Symmetric
label_array_imbalanced_sym = np.copy(labelArray)
sym_noisy_imbalanced_labels = addSymmetricNoise(label_array_imbalanced_sym, 40)

#Balanced Symmetric
label_array_balanced_sym = np.copy(training_labels_balanced_without_noise)
label_array_balanced_sym = addSymmetricNoise(label_array_balanced_sym, 40)

#imbalanced_data = imbalanced_data.reshape(-1, image_size * image_size)
#balanced_data = balanced_data.reshape(-1, image_size * image_size)
#test_images = test_images.reshape(-1, image_size*image_size)
imbalanced_data = imbalanced_data.reshape(-1, 1, image_size, image_size)
balanced_data = balanced_data.reshape(-1, 1, image_size, image_size)
test_images = test_images.reshape(-1, 1, image_size, image_size)

labelArray = np_utils.to_categorical(labelArray, 10)
training_labels_balanced_without_noise = np_utils.to_categorical(training_labels_balanced_without_noise, 10)
label_array_imbalanced_asym = np_utils.to_categorical(label_array_imbalanced_asym, 10)
label_array_balanced_asym = np_utils.to_categorical(label_array_balanced_asym, 10)
label_array_imbalanced_sym = np_utils.to_categorical(label_array_imbalanced_sym, 10)
label_array_balanced_sym = np_utils.to_categorical(label_array_balanced_sym, 10)

test_labelArray = np_utils.to_categorical(test_labelArray, 10)


def get_data():
  return imbalanced_data, labelArray, labelArray, test_images, test_labelArray

def computeAndDisplayConfusionMatrix(test_data, prediction, classifier):
  cm = metrics.confusion_matrix(test_data, prediction, labels=classifier.classes_)
  disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
  disp.plot()
  plt.show()


#LDAM Code Conversions (to torch.tensor)

imbalanced_data_tensor = torch.from_numpy(imbalanced_data)
balanced_data_tensor = torch.from_numpy(balanced_data)

imbalanced_data_labels_tensor = torch.from_numpy(np.argmax(labelArray,axis=1))
balanced_data_labels_tensor = torch.from_numpy(np.argmax(training_labels_balanced_without_noise,axis=1))
imbalanced_data_asym_tensor = torch.from_numpy(np.argmax(label_array_imbalanced_asym,axis=1))
balanced_data_asym_tensor = torch.from_numpy(np.argmax(label_array_balanced_asym,axis=1))
imbalanced_data_sym_tensor = torch.from_numpy(np.argmax(label_array_imbalanced_sym,axis=1))
balanced_data_sym_tensor = torch.from_numpy(np.argmax(label_array_balanced_sym,axis=1))

def getLabel(npArray):
  return list(Counter(np.argmax(npArray,axis=1)).values())

def train_data_tensor(raw = "imbalanced",label = "no"):
  if raw == 'imbalanced':
    if label == "no":
      dataset = torch.utils.data.TensorDataset(imbalanced_data_tensor,imbalanced_data_labels_tensor)
      cls_num_list = getLabel(labelArray)
    elif label == "asym":
      dataset = torch.utils.data.TensorDataset(imbalanced_data_tensor,imbalanced_data_asym_tensor)
      cls_num_list = getLabel(label_array_imbalanced_asym)
    elif label == "sym":
      dataset = torch.utils.data.TensorDataset(imbalanced_data_tensor,imbalanced_data_sym_tensor)
      cls_num_list = getLabel(label_array_imbalanced_sym)
    else:
      print("Incorrect Label")
  elif raw == "balanced":
    if label == "no":
      dataset = torch.utils.data.TensorDataset(balanced_data_tensor,balanced_data_labels_tensor)
      cls_num_list = getLabel(training_labels_balanced_without_noise)
    elif label == "asym":
      dataset = torch.utils.data.TensorDataset(balanced_data_tensor,balanced_data_asym_tensor)
      cls_num_list = getLabel(label_array_balanced_asym)
    elif label =="sym":
      dataset = torch.utils.data.TensorDataset(balanced_data_tensor,balanced_data_sym_tensor)
      cls_num_list = getLabel(label_array_balanced_sym)
    else:
      print("Incorrect Label")
  else:
    print("Incorrect data requested")
  return dataset,cls_num_list

def val_data_tensor():
  test_images_tensor = torch.from_numpy(test_images)
  test_images_label_tensor = torch.from_numpy(np.argmax(test_labelArray,axis=1))
  val_dataset = torch.utils.data.TensorDataset(test_images_tensor,test_images_label_tensor)
  return val_dataset

#print(torch.max(imbalanced_data_tensor))