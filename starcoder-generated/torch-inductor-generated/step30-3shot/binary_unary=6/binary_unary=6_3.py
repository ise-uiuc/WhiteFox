
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
        self.fc = torch.nn.Linear(32, 16)
        self.other = torch.nn.Parameter(torch.ones(32))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        v4 = self.fc(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
