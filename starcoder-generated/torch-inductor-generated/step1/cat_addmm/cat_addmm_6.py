
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(10, 10)
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)
 
    def forward(self, x):
        v1 = self.fc0(torch.randn(10))
        v2 = self.fc1(torch.randn(10))
        v3 = self.fc2(torch.randn(10))
        v4 = self.fc3(torch.randn(10))
        v5 = torch.cat([v1, v2], 0)
        v6 = torch.cat([v3, v4], 0)
        return torch.cat([v5, v6], 1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(12, 10)
