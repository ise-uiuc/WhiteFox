
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(8, 3)
 
    def forward(self, x1, x2):
        v1 = m(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 3)
