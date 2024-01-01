
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu2 = torch.nn.ReLU6(inplace=True)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.rand_like(x1, dtype=torch.float32, device=x1.device)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.relu2(v3)
        return v4
min = 3.4
max = 1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
