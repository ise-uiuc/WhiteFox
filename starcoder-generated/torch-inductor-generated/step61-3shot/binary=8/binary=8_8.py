
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 5, stride=2, padding=2)
    def forward(self, x):
        v = torch.ops.aten.max_pool2d(x, 1, [1], [1])
        v1 = self.conv1(v)
        v2 = self.conv2(v)
        v3 = v1 + v2
        return v3
# Inputs to the model
x = torch.randn(1, 3, 100, 100)
