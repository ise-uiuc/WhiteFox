
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 16)
        self.fc2 = torch.nn.Linear(128, 64)
        
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.cat((v1, x1), dim=1)
        v3 = self.fc2(v2)
        v4 = torch.sigmoid(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 64)
