
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        self.bias = torch.nn.Parameter(torch.arange(16))
        
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.bias
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
