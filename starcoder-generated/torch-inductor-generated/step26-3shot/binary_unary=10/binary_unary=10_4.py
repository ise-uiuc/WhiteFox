
class Model(torch.nn.Module):
    def __init__(self, linear_weight, linear_bias, relu_weight, relu_bias):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.relu = torch.nn.ReLU()
        self.linear.weight = torch.nn.Parameter(linear_weight)
        self.linear.bias = torch.nn.Parameter(linear_bias)
        self.relu.weight = torch.nn.Parameter(relu_weight)
        self.relu.bias = torch.nn.Parameter(relu_bias)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = self.relu(v2)
        v4 = self.relu(v3)
        return v4

# Initializing the model
a = RandomTensor(5, 4, 5, 3)
b = RandomTensor(4, 4, 5)
c = RandomTensor(4, 4, 5)
d = RandomTensor(4, 4, 5)
