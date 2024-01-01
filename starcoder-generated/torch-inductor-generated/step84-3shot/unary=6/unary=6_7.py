
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 6, 3, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = 3 + t2
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        return t7.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
