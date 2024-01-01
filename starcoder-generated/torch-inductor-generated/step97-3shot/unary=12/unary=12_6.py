
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5, stride=1, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(4, 1, 5, stride=1, padding=2, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2.sigmoid()
        v4 = self.dropout(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
