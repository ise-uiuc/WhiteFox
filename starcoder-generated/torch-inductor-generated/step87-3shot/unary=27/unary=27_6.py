
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.min = min
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv2(v2)
        return v3
min = 0
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
