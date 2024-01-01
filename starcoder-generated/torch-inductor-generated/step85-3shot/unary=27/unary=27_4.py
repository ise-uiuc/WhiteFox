
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 2, 3, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = torch.Tensor([[6.7], [7.2], [5.9], [6.7], [8.7], [5.8], [6.7], [7.5], [7.2]])
max = torch.Tensor([[1.6], [1.7], [1.7], [4.9], [2.9], [1.7], [2.7], [3.8], [4.3]])
# Inputs to the model
x1 = torch.randn(1, 6, 224, 224)
