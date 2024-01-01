
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128))
 
    def forward(self, x1):
        v1, _ = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
