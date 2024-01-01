
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1.7
max = 2
# Inputs to the model
x1 = torch.randn(1, 50, 100, 75)
