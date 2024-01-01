
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.181576):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v1)
        v5 = v4 * self.negative_slope
        v6 = torch.where(v4 >= v3, v5, v3)
        return v6
negative_slope = 0.172313
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32, device='cuda:0')
