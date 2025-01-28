import numpy as np
import time
import pandas as pd
import json
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from scipy.stats import randint

from skimage.color import rgb2gray
from skimage.feature import hog

from PIL import Image
from torch import nn
import torchvision.models as models
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


iSecs=time.time()


# Path to Dataset
root_path = r'C:\Users\sicil\OneDrive\Desktop\CS369\IntelTrainingDataset'


# split into subfolders based on class label
subfolders = sorted(glob(root_path + '\\*'))

label_names = [p.split('\\')[-1] for p in subfolders]

label_names = []
for p in subfolders:
  label_names.append(p.split('/')[-1])

img_path= sorted(glob(subfolders[0]+'\\*.jpg'))[0]


ifeat=time.time()

# load the model
resnet50 = models.resnet50(pretrained=True)


# this function gets all the layers of the network
def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])

# this step grabs all but the last layer and transfers it to the cpu
model_conv_features = slice_model(resnet50, to_layer=-1).to('cpu')


# pre-processing steps required by ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# makes sure images are of the correct data type
def retype_image(in_img):
  if np.max(in_img) > 1:
    in_img = in_img.astype(np.uint8)
  else:
    in_img = (in_img * 255.0).astype(np.uint8)
  return in_img

# put the model in evaluation mode
resnet50.eval()

test_img = plt.imread(sorted(glob(subfolders[0] + '/*.jpg'))[0])


proc_img = preprocess(Image.fromarray(retype_image(test_img)))
emb1 = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze()


num_per_class = 20

features = []
labels = []
for i in range(len(subfolders)):
  # get all filepaths
  fnames = sorted(glob(subfolders[i]+'/*.jpg'))

  for j in range(num_per_class):
    img = plt.imread(fnames[j])
    proc_img = preprocess(Image.fromarray(img))
    feat = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()
    features.append(feat)
    labels.append(i)


features = np.array(features)

labels = np.array(labels)

# Define Image Augmentation Transformation
def augment_images(images):
    augmented_images = []
    for img in images:
        if randint(0, 5)== 0:  #  choose whether to flip the image
            img = np.fliplr(img)  # Horizontal flip
        augmented_images.append(img)
    return augmented_images

# Augment Images and Extract Features
augmented_features = []
augmented_labels = []
for i in range(len(subfolders)):
    # Get all filepaths
    fnames = sorted(glob(subfolders[i] + '/*.jpg'))

    # Load images
    images = [plt.imread(fname) for fname in fnames]

    # Augment original images
    augmented_images = augment_images(images)

    for img in augmented_images:
        # Preprocess the image
        proc_img = preprocess(Image.fromarray(retype_image(img)))
        # Extract features
        feat = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()
        augmented_features.append(feat)
        augmented_labels.append(i)

# Concatenate Original and Augmented Features
features = np.concatenate([features, np.array(augmented_features)], axis=0)
labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)


min = str((time.time() - ifeat)/60)
print("Feature vecture data created. Time taken for creation in minutes: " + min)


#split data
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=0,
)

"""
#hyperparameter tuning with cv. Random forests I couldn't get very high test accuracy though. I tried with HOG
param_dist = {'n_estimators': randint(50,200),
              'max_depth': randint(1,15)}

# Create a random forest classifier
rf = RandomForestClassifier()

#find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=20, cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

#I used these returned params for the random forest clf
print('Best hyperparameters:',  rand_search.best_params_)

"""

#make and fit classifer
iClf=time.time()


clf = SVC(kernel='rbf', random_state=42)

clf.fit(X_train, y_train)

min = str((time.time() - iClf)/60)
print("Classifier created and fit to data. Time taken for creation and fitting in minutes: " + min)



# report overall accuracy on the training/test data
iTest = time.time()
TrainScore = clf.score(X_train, y_train)
TestScore =  clf.score(X_test, y_test)
print("Train Accuracy: {}".format(TrainScore))
print('Test Accuracy: {}'.format(TestScore))


# Report accuracy for each class
y_predTest = clf.predict(X_test)
y_predTrain = clf.predict(X_train)


#I couldn't decide wether to put this before or after the confusion matrix,
#but the time wouldn't report until you closed the graph so this seemed to make more sense
min = str((time.time() - iTest)/60)
print("Classifier tested on train and test data. Time taken for scoring in minutes: " + min)

min = str((time.time() - iSecs)/60)
print("Program finished. Minutes since program start: " + min)

# Plot the results as a confusion matrix
Ctrain = confusion_matrix(y_train, y_predTrain)
sn.heatmap(Ctrain, annot=True, cmap='Blues')
plt.title('Train Confusion Matrix Accuracy: ' + str(round(TrainScore,4)))
plt.show()

Ctest = confusion_matrix(y_test, y_predTest)
sn.heatmap(Ctest, annot=True, cmap='Blues')
plt.title('Test Confusion Matrix Accuracy: ' + str(round(TestScore,4)))
plt.show()
