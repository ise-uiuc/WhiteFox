
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 * torch.clamp(torch.nn.functional.relu(v1+3), max=6)
        return v2/6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
