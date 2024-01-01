
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0, 1)
        self.linear = torch.nn.Linear(1020, 1)
        self.clamp = torch.nn.Hardtanh(-1, 3)
    def forward(self, x1):
        v1 = self.flatten(x1)
        v2 = self.linear(v1)
        v3 = self.clamp(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 8, 32, 32)
