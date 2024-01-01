
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(5, stride=2, padding='same', count_include_pad=True)
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.avg_pool(x1)
        v2 = self.avg_pool(x2)
        v3 = self.conv1(v1)
        v4 = self.conv2(v2)
        v5 = v3 + v4
        v6 = self.conv3(v1)
        v7 = self.conv4(v2)
        v8 = v6.sub(v7)
        return v5 * v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
