
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear0 = torch.nn.Linear(7, 9, bias=True)
        self.linear1 = torch.nn.Linear(9, 8, bias=False)
        self.linear2 = torch.nn.Linear(8, 7, bias=True)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v11 = self.linear1(v3)
        v4 = self.linear2(v11)
        return v4
min = -0.5
max = 0.5
# Inputs to the model
x1 = torch.randn(5, 7)
