
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        return self.relu6(self.conv(x1).add(3).clamp_min(0).clamp_max(6).div(6))
# Input to the model
x1 = torch.randn(1, 3, 64, 64)
