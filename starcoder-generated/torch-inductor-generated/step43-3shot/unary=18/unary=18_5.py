
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=160, kernel_size=1, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=320, kernel_size=1, stride=1, padding=0, dilation=5)
        self.conv4 = torch.nn.Conv2d(in_channels=320, out_channels=6400, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=4)
        self.conv5 = torch.nn.Conv2d(in_channels=6400, out_channels=160, kernel_size=1, stride=2, padding=3, dilation=6)
        self.conv6 = torch.nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(1, 6), stride=1, padding=(2, 1), dilation=1)
    def forward(self, inp1):
        v1 = torch.sigmoid(self.conv1(inp1))
        v2 = torch.sigmoid(self.conv2(v1))
        v3 = torch.sigmoid(self.conv3(v2))
        v4 = torch.sigmoid(self.conv4(v3))
        v5 = torch.sigmoid(self.conv5(v4))
        v6 = torch.sigmoid(self.conv6(v5))
        return v6
# Inputs to the model
inp1 = torch.randn(1, 16, 80, 80)
