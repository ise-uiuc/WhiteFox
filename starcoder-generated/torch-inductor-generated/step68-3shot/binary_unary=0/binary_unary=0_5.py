
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
    def forward(self, x):
        t0 = torch.zeros(16, 1, 7, 7, device='cpu')
        t1 = self.conv1(x)
        t2 = x + t1
        t3 = t2 + t0
        t4 = F.conv2d(t3, t0)
        return t4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
