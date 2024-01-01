
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.linear1 = torch.nn.Linear(11, 29)
        self.linear2 = torch.nn.Linear(29, int(self.max) - int(self.min) + 1)
    def forward(self, x1):
        v1 = torch.clamp_min(self.linear1(x1), 3)
        v2 = torch.clamp_max(v1, self.min - 5.5)
        v3 = torch.clamp_min(v2, self.max)
        v4 = self.linear2(v3)
        return v4
min = 1
max = 50
# Inputs to the model
x1 = torch.randn(11)
