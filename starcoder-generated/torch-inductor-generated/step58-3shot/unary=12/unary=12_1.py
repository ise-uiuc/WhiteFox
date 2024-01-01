
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.AvgPool2d(7, stride=1, padding_mode='zeros', include_pad=False)
        self.conv1 = torch.nn.Conv2d(1, 16, (7, 7), stride=1, padding_mode='zeros', groups=1, bias=False)
        self.sigmoid = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.pool1(x1)
        v2 = self.conv1(v1)
        v3 = self.sigmoid(v2)
        v4 = self.pool1(v3)
        v5 = self.conv1(v4)
        v6 = self.sigmoid(v5)
        v7 = v6 * 0.5 + 0.5
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 128, 256)
