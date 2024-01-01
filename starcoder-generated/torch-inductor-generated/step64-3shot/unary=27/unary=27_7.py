
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = x1.view((-1, 39, 60, 1))
        v2 = torch.cat((v1, -v1), dim=0)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        v5 = v4.view((-1, 1, 1480, 3))
        v6 = v5.view((-1, 1480, 2))
        v7 = v6.view((-1, 1480 // 2, 1))
        v8 = v7 + torch.nn.functional.pad(v7, padding=[[None, None], [None, 4], [None, None]])
        return v8
min = 989.9
max = 990.1
# Inputs to the model
x1 = torch.randn(1, 39, 60)
