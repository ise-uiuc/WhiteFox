
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(15, 13, 2, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(13, 11, 5, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * -0.001
        v5 = torch.where(v2 > 0, v2 * -0.001, v2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 15, 21, 21)
# Model begins

