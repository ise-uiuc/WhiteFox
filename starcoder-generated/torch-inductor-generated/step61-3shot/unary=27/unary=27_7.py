
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 60)
        self.linear2 = torch.nn.Linear(60, 10)

        self.min = min
        self.max = max

    def forward(self, x):
        v1 = self.linear1(x)

        v2 = self.linear2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.12
max = 0.22
# Inputs to the model
x = torch.randn(1, 12)
