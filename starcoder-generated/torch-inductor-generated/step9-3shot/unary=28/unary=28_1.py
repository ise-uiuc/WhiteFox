
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 20)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.clamp_min(v1, 0.125)
        v3 = torch.clamp_max(v2, 10.375)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(24, 10, 1, 1)
