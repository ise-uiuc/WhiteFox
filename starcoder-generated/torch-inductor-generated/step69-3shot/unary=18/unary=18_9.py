
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=9)
        self.dropout1 = torch.nn.Dropout2d()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.avg_pool(v2)
        v4 = self.dropout1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(10, 3, 128, 128)
