
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_ = torch.nn.Linear(3, 5, False)
 
    def forward(self, x1):
        v1 = self.linear_(x1)
        v2 = v1 + self.param
        return v2

# Initializing the model
m = Model()
param = torch.randn(5, 3)
m.param = param

# Inputs to the model
x1 = torch.randn(1, 3)
