
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)
 
    def forward(self, x1, x2=None):
        v1 = self.linear(x1)
        v2 = v1 if x2 is None else v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 1, 32)
