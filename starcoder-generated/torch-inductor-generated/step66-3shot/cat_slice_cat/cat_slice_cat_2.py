
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(127, 1024)
        self.fc2 = torch.nn.Linear(1024 + 1, 512)
 
    def forward(self, x1):
        v1 = F.relu(self.fc1(x1))
        v2 = torch.mean(v1, 1)
        v3 = torch.cat([v2, x1], dim=len(x1.shape) - 1)
        v4 = self.fc2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 127)
