
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=32, out_features=64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.rand(64, 32)
        v3 = v1 - v2
        v4 = F.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
