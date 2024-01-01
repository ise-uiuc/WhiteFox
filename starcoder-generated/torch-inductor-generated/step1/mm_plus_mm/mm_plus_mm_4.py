
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input, weights, bias):
        return torch.add(torch.matmul(input, weights), bias)

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(4, 3)
weights = torch.randn(5, 3)
bias = torch.randn(5)
