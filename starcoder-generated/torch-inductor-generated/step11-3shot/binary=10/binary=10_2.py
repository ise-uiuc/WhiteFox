
class Model(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = torch.nn.Linear(param, param)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model with value of "other"
__param__ = ____
m = Model(__param__)

# Inputs to the model
x1 = torch.randn(1, __param__, __param__)
