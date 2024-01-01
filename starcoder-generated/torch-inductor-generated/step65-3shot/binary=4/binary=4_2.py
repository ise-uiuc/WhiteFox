
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1000, out_features=2048, bias=False)
        self.bn = torch.nn.BatchNorm2d(2048)
        self.relu = torch.nn.ReLU()
        self.other = torch.randn(2048, 2048)
 
    def forward(self, x1):
        v0 = x1
        v1 = self.linear(v0)
        v2 = v1 + self.other
        v3 = self.bn(v2)
        v4 = self.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1000)
