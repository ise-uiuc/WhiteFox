
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16, bias=True)
        self.linear.weight = torch.nn.Parameter(weight)
        self.linear.bias = torch.nn.Parameter(bias)
 
    def forward(self, x1, x2):
        v2 = self.linear(x1)
        v3 = v2 + x2
        return v3

# Initializing the model
weight = torch.randn(16, 8)
bias = torch.randn(16)
m = Model(weight, bias)

# Inputs to the model -- the first one is a tensor initialized as an input tensor, and the other one is a tensor not passed from the input tensors of this model.
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 16)
