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

        if creating_objects is not None:
            for child in creating_objects:
                if self.ID not in child.children:
                    child.children[self.ID] = 1
                else:
                    child.children[self.ID] += 1

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

                if self.creating_operations == "sigmoid":
                    ones = Tensor(np.ones_like(self.gradient.data))
                    self.creating_object[0].back_propagation(self.gradient * (self * (ones - self)))

                if self.creating_operations == "tanh":
                    ones = Tensor(np.ones_like(self.gradient.data))
                    self.creating_object[0].back_propagation(self.gradient * (ones - (self * self)))

    def __add__(self, other):
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(self.data + other.data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.multiple_auto_gradient:
            return Tensor(self.data * -1,
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(self.data - other.data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.multiple_auto_gradient and other.multiple_auto_gradient:
            return Tensor(self.data * other.data,
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="mul")
        return Tensor(self.data * other.data)

    def transpose(self):
        if self.multiple_auto_gradient:
            return Tensor(self.data.transpose(),
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="transpose")
        return Tensor(self.data.transpose())

    def sum(self, dimension):
        if self.multiple_auto_gradient:
            return Tensor(self.data.sum(dimension),
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="sum_" + str(dimension))
        return Tensor(self.data.sum(dimension))

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
        if self.multiple_auto_gradient:
            return Tensor(self.data.dot(other.data),
                          multiple_auto_gradient=True,
                          creating_objects=[self, other],
                          creating_operations="scalar_product")
        return Tensor(self.data.dot(other.data))

    def sigmoid(self):
        if self.multiple_auto_gradient:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="sigmoid")

        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.multiple_auto_gradient:
            return Tensor(np.tanh(self.data),
                          multiple_auto_gradient=True,
                          creating_objects=[self],
                          creating_operations="tanh")

        return Tensor(np.tanh(self.data))

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

    def forward(self, input):
        return input.scalar_product(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = list()
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for k in self.layers:
            params += k.get_parameters()
        return params


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input):
        return input.sigmoid()


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predict, goal):
        return ((predict - goal) * (predict - goal)).sum(0)


np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), multiple_auto_gradient=True)
target = Tensor(np.array([[0], [1], [0], [1]]), multiple_auto_gradient=True)
model = Sequential([Dense(2, 3),
                    Tanh(),
                    Dense(3, 1),
                    Sigmoid()])

optim = SGD(parameters=model.get_parameters(), alpha=1)

for i in range(10):
    pred = model.forward(data)
    loss = MSELoss().forward(pred, target)
    loss.back_propagation(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)
