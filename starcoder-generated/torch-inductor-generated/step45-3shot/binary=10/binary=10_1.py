
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)
 
    def forward(self, x):
        v1 = self.features(x)
        v2 = self.bn(v1)
        v3 = v2 + torch.rand(v2.shape)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
