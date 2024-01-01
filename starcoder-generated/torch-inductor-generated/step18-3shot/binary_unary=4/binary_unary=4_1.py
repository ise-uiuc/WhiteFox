
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.linear = torch.nn.Linear(2*16, 1)
 
    def forward(self, x1, t=None):
        v1 = self.fc1(x1)
        v2 = torch.cat([v1, v1], dim=1)
        v3 = self.linear(v2)
        if t is not None:
            v3 = v3 + t
        v4 = F.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
