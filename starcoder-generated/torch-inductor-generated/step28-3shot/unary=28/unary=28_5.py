
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.0, max_value=1.0):
        super().__init__()
        self.fc = torch.nn.Linear(8, 128)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 8)
