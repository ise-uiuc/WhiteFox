
# Some common PyTorch APIs
class Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Weighted sum linear module
class WeightedSumLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
 
    def forward(self, x):
        v = (x * self.weight).sum()
        return v

# Model with weighted sum linear module
class Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear1 = linear
        self.linear2 = linear
        self.linear3 = linear
 
    def forward(self, x1, x2, x3):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = self.linear3(x3)
        v4 = v1 + v2 + v3
        return v4

# Initializing the model (this example uses the weighted sum linear module's forward method as the input parameter of Model's linear argument)
linear = WeightedSumLinear(torch.nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.float32)))
linear.register_parameter("weight", linear.weight) # Avoid the 'AttributeError: 'Parameter' object has no attribute '_grad' when converting the model to ONNX
m = Model(linear)

# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
x2 = torch.randn(1, 2, 3, 3)
x3 = torch.randn(1, 2, 3, 3)
