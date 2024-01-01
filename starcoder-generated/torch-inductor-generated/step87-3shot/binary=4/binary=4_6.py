
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
 
    def forward(self, x1, x2):
        v1 = self.fc1(x1)
        v2 = v1 + x2
        v3 = self.fc2(v2)
        v4 = v3 + x2
        v5 = self.fc3(v4)
        v6 = v5 + x2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
x2 = torch.randn(2, 16)
