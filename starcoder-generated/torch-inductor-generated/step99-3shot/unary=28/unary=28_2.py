
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 1)
        v4 = v1 + v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 10)
