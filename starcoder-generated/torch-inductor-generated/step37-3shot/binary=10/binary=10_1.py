
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(8))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.bias
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
