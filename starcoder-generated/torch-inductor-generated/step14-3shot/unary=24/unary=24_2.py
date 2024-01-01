
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.nn.functional.pixel_shuffle(v1, 4)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.pixel_shuffle(v3, 4)
        v5 = self.conv3(v4)
        return v5
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
