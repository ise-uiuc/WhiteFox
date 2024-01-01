
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v = self.m(x1)
        return v + x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 3)
