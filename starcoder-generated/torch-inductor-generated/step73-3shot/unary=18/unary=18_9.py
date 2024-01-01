
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
    def forward(self, x1):
        v1 = self.conv2(torch.sigmoid(self.conv1(x1)))
        v2 = self.conv4(torch.sigmoid(self.conv3(v1)))
        v3 = torch.sigmoid(self.conv5(v2))
        return v3
# Inputs to the model
x3 = torch.randn(1, 3, 224, 224)
