
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(896, 96)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min=0)
        v3 = torch.clamp_max(v2, max=0.5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 896)
