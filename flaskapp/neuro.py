import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.keras import datasets, layers, models

def neuro(imge):

    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images / 255, testing_images /255

    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    for i in range(16):
        plot.subplot(4,4,i+1)
        plot.xticks([])
        plot.yticks([])
        plot.imshow(training_images[i], cmap = plot.cm.binary)
        plot.xlabel(class_names[training_labels[i][0]])

    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]
 
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model = models.load_model('image_classifier.model')

    img = cv.imread(imge)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plot.imshow(img, cmap=plot.cm.binary)

    prediction = model.predict(np.array([img])/ 255)
    index = np.argmax(prediction)
    return(class_names[index])