
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(576, 128)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, start_dim=1)
        v3 = self.linear(v2)
        v4 = torch.clamp(v3 + 3, 0, 6) 
        v5 = v4 / 6 
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
