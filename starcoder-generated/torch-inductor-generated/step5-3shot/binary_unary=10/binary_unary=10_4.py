
class Model(torch.nn.Module):
    def __init__(self, __other__):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + __other__
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(__other__=__other__)

# Inputs to the model
x1 = torch.randn(1, 64)
