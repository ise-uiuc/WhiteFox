
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn1 = torch.nn.Linear(64, 32)
 
    def forward(self, x):
        v1 = self.nn1(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
