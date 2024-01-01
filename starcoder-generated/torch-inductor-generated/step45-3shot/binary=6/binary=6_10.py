
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, x1, param):
        v1 = self.linear(x1)
        v2 = v1 - param
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
param = torch.randn(32)
