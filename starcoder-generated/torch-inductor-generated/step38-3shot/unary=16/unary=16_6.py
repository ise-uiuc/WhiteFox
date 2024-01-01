
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64,8)
 
    def forward(self, x2):
        v2 = self.fc(x2)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
n = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
