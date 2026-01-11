class mnist_datasets:

    def __init__(self, normalize=True, flatten=True):
        self.name = "MNIST"
        self.normalize = normalize
        self.flatten = flatten

    def load_idx_images(self, filename):
        import numpy as np
        import struct

        with open(filename, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)

        if self.flatten:
            images = images.reshape(n, rows * cols)
        else:
            images = images.reshape(n, rows, cols)

        if self.normalize:
            images = images.astype(np.float32) / 255.0

        return images

    def load_idx_labels(self, filename):
        import numpy as np
        import struct

        with open(filename, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

    def load_data(self, train=True):
        if train:
            X = self.load_idx_images("C:\\Users\\farza\\Downloads\\archive\\train-images.idx3-ubyte")
            y = self.load_idx_labels("C:\\Users\\farza\\Downloads\\archive\\train-labels.idx1-ubyte")
        else:
            X = self.load_idx_images("C:\\Users\\farza\\Downloads\\archive\\t10k-images.idx3-ubyte")
            y = self.load_idx_labels("C:\\Users\\farza\\Downloads\\archive\\t10k-labels.idx1-ubyte")

        return X, y


if __name__ == "__main__":
    dataset=mnist_datasets()
    X_train, y_train = dataset.load_data(train=True)
    X_test, y_test = dataset.load_data(train=False)

    print(f"shape of X_train: {X_train.shape}")
    print(f"shape of y_train: {y_train.shape}")
    print(f"shape of X_test: {X_test.shape}")
    print(f"shape of y_test: {y_test.shape}")