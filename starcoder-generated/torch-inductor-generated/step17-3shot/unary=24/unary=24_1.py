
# For this section, feel free to reference the Leaky-ReLU activation function as it is part of most standard neural network frameworks.
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
    def forward(self, x):
        v1 = self.leaky_relu(x)
        return v1
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
