
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64*64*3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = self.fc2(v1)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64*64*3)
