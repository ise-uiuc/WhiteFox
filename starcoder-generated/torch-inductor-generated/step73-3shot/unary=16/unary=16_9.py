
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = F.linear(x1, weight, bias)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Generating the weights
weight = torch.randn(64, 128)
bias = torch.randn(128)

# Inputs to the model
x1 = torch.randn(1, 64)
