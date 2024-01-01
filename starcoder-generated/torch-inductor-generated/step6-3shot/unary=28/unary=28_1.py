
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 2)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min=x2)
        v3 = torch.clamp_max(v2, max=x2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1)
