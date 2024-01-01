
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=2)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=3)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 17, 1, requires_grad=True)
