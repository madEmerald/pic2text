import numpy as np


class Tensor(object):
    def __init__(self, data,
                 multiple_auto_gradient=False,
                 creating_objects=None,
                 creating_operations=None,
                 ID=None):

        self.data = np.array(data)
        self.creating_object = creating_objects
        self.creating_operations = creating_operations
        self.gradient = None
        self.multiple_auto_gradient = multiple_auto_gradient
        self.children = {}
        if ID is None:
            ID = np.random.randint(0, 1000000000)
        self.ID = ID

        self.softmax_output = None
        self.target_dist = None
        self.dropout_mask = None

        if creating_objects is not None:
            for child in creating_objects:
                if self.ID not in child.children:
                    child.children[self.ID] = 1
                else:
                    child.children[self.ID] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def all_children_gradient_accounted_for(self):
        for ID, count in self.children.items():
            if count != 0:
                return False
        return True

    def back_propagation(self, gradient=None, gradient_original=None):
        if self.multiple_auto_gradient:

            if gradient is None:
                gradient = Tensor(np.ones_like(self.data))

            if gradient_original is not None:
                if self.children[gradient_original.ID] == 0:
                    raise Exception("cannot back more than once")
                else:
                    self.children[gradient_original.ID] -= 1

            if self.gradient is None:
                self.gradient = gradient
            else:
                self.gradient += gradient

            if self.creating_object is not None and (self.all_children_gradient_accounted_for() or
                                                     gradient_original is None):
                if self.creating_operations == "add":
                    self.creating_object[0].back_propagation(self.gradient, self)
                    self.creating_object[1].back_propagation(self.gradient, self)

                if self.creating_operations == "neg":
                    self.creating_object[0].back_propagation(self.gradient.__neg__())

                if self.creating_operations == "transpose":
                    self.creating_object[0].back_propagation(self.gradient.transpose())

                if self.creating_operations == "sub":
                    new = Tensor(self.gradient.data)
                    self.creating_object[0].back_propagation(new, self)
                    new = Tensor(self.gradient.__neg__().data)
                    self.creating_object[1].back_propagation(new, self)

                if self.creating_operations == "mul":
                    new = self.gradient * self.creating_object[1]
                    self.creating_object[0].back_propagation(new, self)
                    new = self.gradient * self.creating_object[0]
                    self.creating_object[1].back_propagation(new, self)

                if self.creating_operations == "scalar_product":
                    activation_layer = self.creating_object[0]
                    weights_layer = self.creating_object[1]
                    new = self.gradient.scalar_product(weights_layer.transpose())
                    activation_layer.back_propagation(new)
                    new = self.gradient.transpose().scalar_product(activation_layer).transpose()
                    weights_layer.back_propagation(new)

                if "sum" in self.creating_operations:
                    dimension = int(self.creating_operations.split("_")[1])
                    dimension_side = self.creating_object[0].data.shape[dimension]
                    self.creating_object[0].back_propagation(self.gradient.expand(dimension, dimension_side))

                if "expand" in self.creating_operations:
                    dimension = int(self.creating_operations.split("_")[1])
                    self.creating_object[0].back_propagation(self.gradient.sum(dimension))

                if self.creating_operations == "dropout":
                    self.creating_object[0].back_propagation(self.gradient * self.dropout_mask)

                if self.creating_operations == "sigmoid":
                    ones = Tensor(np.ones_like(self.gradient.data))
                    self.creating_object[0].back_propagation(self.gradient * (self * (ones - self)))

                if self.creating_operations == "tanh":
                    ones = Tensor(np.ones_like(self.gradient.data))
                    self.creating_object[0].back_propagation(self.gradient * (ones - (self * self)))

                if self.creating_operations == "relu":
                    self.creating_object[0].back_propagation(self.gradient * np.maximum(self, 0))

                if self.creating_operations == "cross_entropy":
                    self.creating_object[0].back_propagation(Tensor(self.softmax_output - self.target_dist))

                if self.creating_operations == "linear":
                    return

    def __add__(self, other):
        data = self.data + other.data
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="add")
        return Tensor(data)

    def __neg__(self):
        data = self.data * -1
        if self.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="neg")
        return Tensor(data)

    def __sub__(self, other):
        data = self.data - other.data
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="sub")
        return Tensor(data)

    def __mul__(self, other):
        data = self.data * other.data
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="mul")
        return Tensor(data)

    def transpose(self):
        data = self.data.transpose()
        if self.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="transpose")
        return Tensor(data)

    def sum(self, dimension):
        data = self.data.sum(dimension)
        if self.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="sum_" + str(dimension))
        return Tensor(data)

    def expand(self, dimension, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dimension, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.multiple_auto_gradient:
            return Tensor(new_data,
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="expand_" + str(dimension))
        return Tensor(new_data)

    def scalar_product(self, other):
        data = self.data.dot(other.data)
        if self.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="scalar_product")
        return Tensor(data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for param in self.parameters:
            param.gradient.data *= 0

    def step(self, zero=True):
        for param in self.parameters:
            param.data -= param.gradient.data * self.alpha
            if zero:
                param.gradient.data *= 0


class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Dense(Layer):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        weight_preparation = np.random.randn(num_inputs, num_outputs) * np.sqrt(2.0 / num_inputs)
        self.weight = Tensor(weight_preparation, multiple_auto_gradient=True)
        self.bias = Tensor(np.zeros(num_outputs), multiple_auto_gradient=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input_layer):
        return input_layer.scalar_product(self.weight) + self.bias.expand(0, len(input_layer.data))


class Sequential(Layer):
    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = list()
        self.layers = layers

    def forward(self, input_layer):
        for layer in self.layers:
            input_layer = layer.forward(input_layer)
        return input_layer

    def get_parameters(self):
        params = list()
        for k in self.layers:
            params += k.get_parameters()
        return params


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input_layer):
        mask = (np.random.rand(*input_layer.data.shape) > self.p) / (1.0 - self.p)
        data = input_layer.data * mask
        if input_layer.multiple_auto_gradient:
            out = Tensor(data,
                         multiple_auto_gradient=True,
                         creating_objects=[input_layer],
                         creating_operations="dropout")
            out.dropout_mask = mask
            return out
        return Tensor(data)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_layer):
        data = np.tanh(input_layer.data)
        if input_layer.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[input_layer],
                          creating_operations="tanh")

        return Tensor(data)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_layer):
        data = 1 / (1 + np.exp(-input_layer.data))
        if input_layer.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[input_layer],
                          creating_operations="sigmoid")

        return Tensor(data)


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    @staticmethod
    def forward(input_layer):
        data = np.maximum(input_layer.data, 0)
        if input_layer.multiple_auto_gradient:
            return Tensor(data,
                          multiple_auto_gradient=True,
                          creating_objects=[input_layer],
                          creating_operations="relu")

        return Tensor(data)


class Linear(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_layer):
        if input_layer.multiple_auto_gradient:
            return Tensor(input_layer.data,
                          multiple_auto_gradient=True,
                          creating_objects=[input_layer],
                          creating_operations="linear")

        return Tensor(input_layer.data)


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict, target):
        return ((predict - target) * (predict - target)).sum(0)


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict, target):
        exp = np.exp(predict.data)
        softmax_output = exp / np.sum(exp)

        loss = -(np.log(softmax_output) * target).sum(1)
        if predict.multiple_auto_gradient:
            out = Tensor(loss,
                         multiple_auto_gradient=True,
                         creating_objects=[predict],
                         creating_operations="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target
            return out

        return Tensor(loss)


class Network:
    def __init__(self, data, target, model, alpha,
                 loss_function=None, test_data=None, test_target=None):
        if not isinstance(data, Tensor):
            data = Tensor(data)
        data.multiple_auto_gradient = True

        if not isinstance(target, Tensor):
            target = Tensor(target)
        target.multiple_auto_gradient = True

        if test_data is not None:
            if not isinstance(test_data, Tensor):
                test_data = Tensor(test_data)
            test_data.multiple_auto_gradient = True

        if test_target is not None:
            if not isinstance(test_target, Tensor):
                test_target = Tensor(test_target)
            test_target.multiple_auto_gradient = True

        self.data = data
        self.target = target

        self.test_data = test_data
        self.test_target = test_target

        self.model = model
        self.loss_function = loss_function if loss_function is not None else MSELoss

        self.optim = SGD(parameters=model.get_parameters(), alpha=alpha)

    def learning(self, num_epochs, test_frequency=0):
        for i in range(1, num_epochs + 1):
            predict = self.model.forward(self.data)

            loss = self.loss_function().forward(predict, self.target)
            loss.back_propagation(Tensor(np.ones_like(loss.data)))
            self.optim.step()

            print("Iteration:", i, "\tTrain error:", loss.data.mean() / len(self.data), end='\t')

            if test_frequency and i % test_frequency == 0:
                self.testing(self.test_data, self.test_target)

            print()

    def testing(self, data, target):
        if data is None or target is None:
            return

        pred = self.model.forward(data)
        n_correct = 0

        for i in range(len(pred)):
            if np.argmax(pred[i]) == np.argmax(target[i]):
                n_correct += 1

        print("Test accuracy:", n_correct / len(data), end='')


def main():
    np.random.seed(0)

    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), multiple_auto_gradient=True)
    target = Tensor(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), multiple_auto_gradient=True)
    model = Sequential([Dense(2, 3),
                        Tanh(),
                        Dense(3, 1)])

    net = Network(data, target, model, 1, loss_function=CrossEntropyLoss)
    net.learning(10)


if __name__ == "__main__":
    main()
