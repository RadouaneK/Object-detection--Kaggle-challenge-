import numpy as np
import cv2
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(600,600), n_channels=3,
                  grid_cells=(19,19), anchor_boxes=5, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.grid_cells = grid_cells
        self.anchor_boxes = anchor_boxes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.grid_cells, self.anchor_boxes, 5+self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load, preproces and store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            img = cv2.imread('data/' + ID)
            img = cv2.resize(img, self.dim)
            img /= 255.
            #img = np.expand_dims(img, 0)
            X[i,] = img
            # Store class
            y[i,] = generate_labels(self.labels[ID], self.grid_cells, 
                                    self.anchor_boxes, self.n_classes)
        return X, y 

def generate_labels(image_annotations, grid_cells, box, n_classes):

    y = np.zeros((*grid_cells, box, 5 + n_classes))

    anchor_index  = 1 # an index for the anchor box
    for annotation in image_annotations:
        index_x = int(annotation[0] * grid_cells[0])
        index_y = int(annotation[1] * grid_cells[1])

        if y[index_x, index_y,:, 4] == 0:
            y[index_x, index_y,:, 0:4]  = annotation[0:4]
            y[index_x, index_y,:, 4] = 1.
            y[index_x, index_y,:, 5:] = keras.utils.to_categorical(annotation[4], num_classes=n_classes)[0]

        else:
            y[index_x, index_y, anchor_index, 0:4]  = annotation[0:4]
            y[index_x, index_y, anchor_index, 4] = 1.
            y[index_x, index_y, anchor_index, 5:] = keras.utils.to_categorical(annotation[4], num_classes=n_classes)[0]

            anchor_box += 1 
    return y 

#y = generate_labels((13,13), 5, 10)
#print(y.shape)

