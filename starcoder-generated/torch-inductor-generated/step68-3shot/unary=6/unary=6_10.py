
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 13, 10, stride=10)
        self.pool = torch.nn.AvgPool1d(5)
        self.act = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.pool(v6)
        v8 = self.act(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 640)
