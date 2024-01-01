
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 2, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(2, 1, 2, stride=1, padding=1)
        self.softmax1 = torch.nn.LogSoftmax(dim=-1)
        self.softmax2 = torch.nn.LogSoftmax(dim=-1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.softmax1(v3)
        v5 = self.softmax2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
