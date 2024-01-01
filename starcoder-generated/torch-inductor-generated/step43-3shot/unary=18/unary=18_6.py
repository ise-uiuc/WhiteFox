
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv4(v5)
        v7 = torch.dropout(v6, p=0.3)
        return torch.sigmoid(v7)
# Inputs to the model
x1 = torch.randn(1, 32, 224, 224)
