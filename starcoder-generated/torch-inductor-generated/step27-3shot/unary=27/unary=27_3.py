
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.add = torch.nn.Add()
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.add(x1, x2)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1.5
max = -2.2
# Inputs to the model
x1 = torch.randn(1, 64, 64, 32)
x2 = torch.randn(1, 32, 32, 32)
