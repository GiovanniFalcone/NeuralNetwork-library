import numpy as np

class Dataset:
    @staticmethod
    def load_mnist():
        """
        This function load the MNIST dataset and return training and test sets.

        Returns:
        ---------
            train_imgs (matrix): The matrix of digits for training. The matrix has a shape of (784, 60000) where 784 is the number of pixel (feature)
            and 60000 is the number of samples.
            train_labels (matrix): The one-hot encoded matrix, which has a shape of (10, 60000) where 10 is the number of classes and 60000 is the number of
            samples.
            test_imgs (matrix): The matrix of digit images for test. The matrix has a shape of (784, 10000).
            test_labels (matrix): The one-hot encoded matrix of labels used for testing. It has a shape of(10, 10000).
        """
        # get all dataset 
        train_data = np.loadtxt("./dataset/mnist_train.csv", delimiter=",")
        test_data = np.loadtxt("./dataset/mnist_test.csv", delimiter=",")
        # in order to have [0, 1] range
        normalization_coefficient = 1 / 255
        # get data
        train_imgs = np.asfarray(train_data[:, 1:]) * normalization_coefficient
        test_imgs = np.asfarray(test_data[:, 1:]) * normalization_coefficient
        # get labels
        train_labels = np.asfarray(train_data[:, :1])
        test_labels = np.asfarray(test_data[:, :1])
        # one-hot label
        c_classes = 10
        lr = np.arange(c_classes)
        train_labels_one_hot = (lr == train_labels).astype(np.int32)
        test_labels_one_hot = (lr == test_labels).astype(np.int32)

        return train_imgs.transpose(), train_labels_one_hot.transpose(), test_imgs.transpose(), test_labels_one_hot.transpose()
    
    @staticmethod
    def train_val_split(data, target, percentage=0.2, random_state=None, shuffle=True):
        """
        This function split the training set in two different sets: training set and validation set.

        Attributes:
        ----------
            data (matrix): the training set organized as matrix (d, N) where d=number of features and N=number of samples.
            target (matrix): The one-hot matrix used for training.
            percentage (float): The percentage used to split (e.g 80% for training and 20% for validation).
            Default is 20%.
            random_state (int): Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls. Default is None
            shuffle (bool): Whether or not to shuffle the data before splitting. Default is True.

        Returns:
        ---------
            x_train (matrix): the new training of images set with a shape of (784, 60000-percentage) where 784 is the number of pixel.
            y_train (matrix): the new one-hot encoded matrix of labels with a shape of (10, 60000-percentage) where 10 is the number of target classes.
            x_val (matrix): the validation set of images with a shape of (784, percentage).
            y_val (matrix): the one-hot encoded matrix of labels with a shape of (10, percentage).

        Errors:
        --------
            ValueError if data or target aren't numpy arrays or if their shape is not 2.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise ValueError(f"Data must be a numpy array with a shape of (d, N)!")
        if not isinstance(target, np.ndarray) or len(data.shape) != 2:
            raise ValueError(f"Target must be a numpy array with a shape of (c, N)!")

        if random_state is not None:
            np.random.seed(random_state)

        # get size of new training set
        split = int((1 - percentage) * data.shape[1])

        # shuffle data before splitting
        if shuffle:
            random_permutation = np.random.permutation(data.shape[1])
            # shuffle samples
            data = data[:, random_permutation]
            target = target[:, random_permutation]
        
        # split data into training and validation set
        x_train, x_val = data[:, :split], data[:, split:]
        y_train, y_val = target[:, :split], target[:, split:]

        return x_train, y_train, x_val, y_val
            