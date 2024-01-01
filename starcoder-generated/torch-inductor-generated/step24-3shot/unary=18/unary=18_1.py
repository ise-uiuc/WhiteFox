
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        dout = torch.nn.Dropout(p=0.3)
        m = dout(v1)
        v2 = torch.sigmoid(m)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv4(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
