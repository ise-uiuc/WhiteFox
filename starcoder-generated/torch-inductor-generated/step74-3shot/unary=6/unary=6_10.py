
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(8, 10, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(10, 10, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_max(v2, 6)
        v4 = torch.clamp_max(v3, 9)
        v5 = v1 * v4
        v6 = v5 * 3
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 128)
