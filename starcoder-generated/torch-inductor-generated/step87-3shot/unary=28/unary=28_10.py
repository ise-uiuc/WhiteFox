
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.25, max_value=0.5):
        super().__init__()
        self.fc = torch.nn.Linear(30, 10)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min_value=-0.25)
        v3 = torch.clamp_max(v2, max_value=0.5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30)
