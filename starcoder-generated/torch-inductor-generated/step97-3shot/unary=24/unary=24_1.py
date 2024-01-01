
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(34, 15, 1, stride=2, padding=[[0, 0], [1, 1], [0, 0], [0, 0]])
        self.conv2 = torch.nn.Conv2d(15, 56, 1, stride=1, padding=[0, 1, 0, 0])
    def forward(self, x):
        negative_slope = 4.670708
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 34, 29, 88)
