
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv1 = torch.nn.Conv2d(1, 1, 5, stride=(1, 1), bias=True, padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=(1, 1), bias=True, padding=(1, 1))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0
max = 1
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
