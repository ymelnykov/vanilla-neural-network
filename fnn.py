"""
Implement a feedforward neural network (FNN)
with a backpropagation method to train it.
"""

import random

class Value:
    """
    Present an object to store 
    a single scalar value and its gradient.
    
    Attributes:
        data (int, float): Scalar value
        gradient (int, float): Gradient value
        children (tuple[Value]): Child nodes in the 
            propagation graph
    
    Methods:
        relu: Apply a REctified Linear Unit activation function
            to avoid linearity
        backward: Implement a backward propagation
    """

    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0
        # private variables used for propagation graph construction
        self._prev = children
        self._backward = lambda: None

    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other): # raise self to the power of other
        # Raise AssertionError if other is not of integer or float type
        assert isinstance(other, (int, float)), 'Power must be of integer or float type'
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        Apply a REctified Linear Unit activation function
        to avoid linearity.
        """

        out = Value(0 if self.data<0 else self.data, (self,))

        def _backward():
            self.grad += (out.data>0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Implement a backward propagation.
        """
        topo = []
        visited = set()

        def build_topo(curr):
            """
            Build a list of propagation graph nodes
            (variables) in topological order.

            Args:
                curr (Value): A propagation graph node

            Returns:
                topo(list[Value]): A list of propagation
                    graph nodes in topological order
            """
            if curr not in visited:
                visited.add(curr)
                for child in curr._prev:
                    build_topo(child)
                topo.append(curr)

        build_topo(self)

        self.grad = 1
        # Get a gradiaent of each node (variable)
        # starting from the end of topo list and
        # using a chain rule
        for curr in reversed(topo):
            curr._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f'Value(data={self.data}, gradient={self.grad})'


class Module:
    """
    Present a neural network module
    (neuron, layer, network).
    """

    def parameters(self):
        """Return a list of module parameters."""
        return []

    def zero_grad(self):
        """
        Initialize all parameter gradients 
        to zero after each iteration.
        """
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    """
    Present a neuron of the neural network.
    
    Attributes:
        num_inputs (int): Number of inputs to the neuron
        nonlin (bool): Flag for a non-linear output
            (default is True)
        weights (list[float]): List of weights for
            inputs. The weights are randomly initialized
        bias (float): Bias. It's randomly initialized

    Methods:
        parameters: Return a list of the neuron parameters 
            (weights and bias)
    """

    def __init__(self, num_inputs, nonlin=True):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.nonlin = nonlin

    def __call__(self, inputs):
        out = sum((weight*input for weight, input in zip(self.weights, inputs)), self.bias)
        return out.relu() if self.nonlin else out

    def parameters(self):
        """Return a list of the neuron parameters (weights and bias)."""
        return self.weights + [self.bias]

    def __repr__(self):
        return f'{"ReLu" if self.nonlin else "Linear"} Neuron({len(self.weights)})'


class Layer(Module):
    """
    Present a layer of the neural network.

    Attributes:
        num_inputs (int): Number of inputs to the layer
        num_outputs (int): Number of neurons in the layer
        neurons (list[Neuron]): List of neurons in the layer

    Methods:
        parameters: Return a list of the layer parameters
            (weights and biases)
    """

    def __init__(self, num_inputs, num_outputs, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, inputs):
        out = [neuron(inputs) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Return a list of the layer parameters (weights and biases)."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f'Layer of [{", ".join(str(neuron) for neuron in self.neurons)}]'


class FNN(Module):
    """
    Present a Feedforward Neural Network (FNN)
    neural network.

    Attributes:
        num_inputs (int): Number of inputs to the FNN
        nums_outputs (list[int]): List of numbers of neurons
            in the FNN layers
        layers (list[Layer]): List of layers in the FNN

    Methods:
        parameters: Return a list of the FNN parameters
            (weights and biases)
    """

    def __init__(self, num_inputs, nums_outputs):
        size = [num_inputs] + nums_outputs
        # activation function should not be applied
        # to the last layer
        self.layers = [Layer(size[i], size[i+1],
                             nonlin = i!=(len(nums_outputs)-1))
                          for i in range(len(nums_outputs))]

    def __call__(self, in_out):
        for layer in self.layers:
            in_out = layer(in_out)
        return in_out

    def parameters(self):
        """Return a list of the FNN parameters (weights and biases)"""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f'FNN of [{", ".join(str(layer) for layer in self.layers)}]'
    