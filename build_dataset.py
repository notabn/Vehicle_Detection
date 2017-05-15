import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from Vehicle_Detection.lesson_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from Vehicle_Detection.lesson_functions import *
import time
import pickle
import collections
import matplotlib.gridspec as gridspec


import sys

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

car_images = glob.glob('vehicles/*/*.png')
notcar_images = glob.glob('non-vehicles/*/*.png')
cars = []
notcars = []

for car,notcar in zip(car_images,notcar_images):
    cars.append(car)
    notcars.append(notcar)


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    img = cv2.imread(notcar_list[0])
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict


data_info = data_look(cars, notcars)


print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])


# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# performs under different binning scenarios
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32,32 ) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
#y_start_stop = [None, None] # Min and max in y to search in slide_window()

def generate_visualization():
    features_car, hog_ch1_car,spatial_features_car = extract_features_plot(car_image, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=0, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    features_notcar, hog_ch1_notcar,spatial_features_nocar = extract_features_plot(notcar_image, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=0, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    fig = plt.figure(figsize=(6, 9))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.6, hspace=0.3)
    plt.subplot(gs1[0])
    plt.imshow(car_image[:,:,0],cmap='gray')
    plt.title('Car CH-1')
    plt.subplot(gs1[1])
    plt.imshow(hog_ch1_car,cmap='gray')
    plt.title('Car CH-1 HOG')
    plt.subplot(gs1[2])
    plt.imshow(notcar_image[:,:,0],cmap='gray')
    plt.title('not Car CH-1')
    plt.subplot(gs1[3])
    plt.imshow(hog_ch1_notcar,cmap='gray')
    plt.title('not Car CH-1 HOG')
    i = 4
    for j in range(0,3):
        plt.subplot(gs1[4*j+i])
        plt.imshow(car_image[:,:,j],cmap='gray')
        plt.title('Car CH-'+str(j))
        plt.subplot(gs1[4*j+1+i])
        plt.imshow(features_car[:,:,j],cmap='gray')
        plt.title('Car CH-'+str(j)+' Features')
        plt.subplot(gs1[4*j+2+i])
        plt.imshow(notcar_image[:,:,j],cmap='gray')
        plt.title('not Car CH-'+str(j))
        plt.subplot(gs1[4*j+3+i])
        plt.imshow(features_notcar[:,:,j],cmap='gray')
        plt.title('Car CH-'+str(j)+'Features')


    fig.savefig('output_images/HOG_example')

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    fig.savefig('output_images/car_notcar')





car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


print('feature extraction done')

if len(car_features) < 0:
    print('Your function only returns empty feature vectors...')
    sys.exit()

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)


# Plot an example of raw and scaled features
car_ind = np.random.randint(0, len(cars))
fig = plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(mpimg.imread(cars[car_ind]))
plt.title('Original Image')
plt.subplot(132)
plt.plot(X[car_ind])
plt.title('Raw Features')
plt.subplot(133)
plt.plot(scaled_X[car_ind])
plt.title('Normalized Features')
fig.tight_layout()
fig.savefig('output_images/Normalized Features')



# Define a labels vector based on features lists
y = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 5))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins

pickle._dump(dist_pickle,open("svc_pickle.p", "wb"))

print('done')



