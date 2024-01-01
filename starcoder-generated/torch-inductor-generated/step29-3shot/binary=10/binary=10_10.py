
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 16)
 
    def forward(self, x1, param):
        v1 = self.linear(x1)
        v2 = v1 + param
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
param = torch.tensor(1.23)
