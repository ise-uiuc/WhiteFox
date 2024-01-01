
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=11, stride=4, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        return nn.ReLU()(v3)
# Inputs to the model
x1 = torch.randn(1, 16, 23, 23)
