
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, 2, stride=8, padding=6)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2,self.max)
        return v3
min = 1.8
max = 2.4
# Inputs to the model
x1 = torch.randn(1, 2, 48).unsqueeze(-1)
