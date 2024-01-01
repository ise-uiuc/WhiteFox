
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
        )
 
    def forward(self, x1):
        v1 = self.layers(x1)
        v2 = v1 * torch.clamp(torch.min(v1), 0, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
