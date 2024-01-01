
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        v1 = self.t(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
