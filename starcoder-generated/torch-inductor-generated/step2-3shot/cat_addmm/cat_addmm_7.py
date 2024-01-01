
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(12, 10)
        self.fc2 = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = torch.cat([v1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 12)
