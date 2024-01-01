
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.tanh = torch.nn.Tanh()
 
    def forward(self, x):
        x = torch.nn.functional._linear.default(x, self.weight, self.bias)
        x = self.tanh(x)
        return x

# Initializing the model
weight = torch.rand(3, 6)
bias = torch.rand(6)
m = Model(weight, bias)

# Inputs to the model
x = torch.randn(2, 3)
