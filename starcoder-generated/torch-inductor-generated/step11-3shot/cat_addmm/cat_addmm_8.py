
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 200)
        self.fc2 = torch.nn.Linear(200, 50)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = v2.unsqueeze(0)
        v4 = torch.cat([v2.unsqueeze(1), v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
