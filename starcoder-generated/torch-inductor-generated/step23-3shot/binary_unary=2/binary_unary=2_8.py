
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5, stride=2, padding=2)
        self.pool1 = torch.nn.AvgPool2d(2, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 16, 5, stride=2, padding=2)
        self.pool2 = torch.nn.AvgPool2d(2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pool1(v1)
        v3 = v2 - 0.5
        v4 = self.conv2(v3)
        v5 = self.pool2(v4)
        v6 = v5 - 1
        v7 = F.relu(v6)
        v8 = torch.squeeze(v7, 0)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
