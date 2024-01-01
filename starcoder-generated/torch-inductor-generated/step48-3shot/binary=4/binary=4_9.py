
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, v1, v2):
        v3 = self.linear(v1)
        v4 = v3 + v2
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__v1__ = torch.randn(1, 128)
__v2__ = torch.randn(1, 128)
