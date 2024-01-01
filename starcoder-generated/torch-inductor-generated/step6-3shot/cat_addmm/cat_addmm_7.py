
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 8)
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.fc2(v1)
        v3 = torch.addmm(v2, v1, v2)
        return torch.cat([v2, v3], 1)

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(32, 32)
