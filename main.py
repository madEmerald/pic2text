from keras.datasets import mnist

DATASET_SIZE = 10


def get_image(data):
    data = data / 255
    return data[:DATASET_SIZE].reshape(DATASET_SIZE, 28 * 28)


def get_labels(data):
    return [[1 if j == data[i] else 0 for j in range(10)] for i in range(DATASET_SIZE)]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_image = get_image(x_train)
train_labels = get_labels(y_train)

test_image = get_image(x_test)
test_labels = get_labels(y_test)
