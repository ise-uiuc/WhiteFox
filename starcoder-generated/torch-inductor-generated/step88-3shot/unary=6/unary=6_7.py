
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=4)
    def forward(self, x1, x2):
        t1 = self.conv1(x1)
        t2 = x2 + 3
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        v1 = self.conv2(t5)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 20)
x2 = torch.randn(2, 3, 28, 28)
