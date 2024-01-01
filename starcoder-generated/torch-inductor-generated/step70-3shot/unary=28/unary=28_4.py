
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(40, 30)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        return torch.clamp_min(v1, min_value=-1.0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 40)
