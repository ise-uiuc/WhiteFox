

class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.model = torch.nn.Linear(10, 10)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.model(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.5
max = 0.5
x1 = torch.randn(512, 10)

