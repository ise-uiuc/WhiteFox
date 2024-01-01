
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.clamp_min(x1[0], self.min)
        v2 = torch.clamp_max(v1, self.max)
        v3 = torch.clamp_min(x1[1], self.min)
        v4 = torch.clamp_max(v3, self.max)
        v5 = torch.clamp_min(x1[2], self.min)
        v6 = torch.clamp_max(v5, self.max)
        out = torch.cat((v2, v4, v6), 0)
        return out
min = 1.0
max = 1.0
# Inputs to the model
x1 = torch.randn(3, 128, 93)
