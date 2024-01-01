
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + 3.0
        v3 = torch.clamp(v2, 0.0, 6.0)
        v4 = v3 / 6.0
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
