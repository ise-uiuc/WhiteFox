
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = torch.sigmoid(self.max_pool(self.conv1(x1)))
        v2 = torch.sigmoid(self.conv3(self.conv2(v1)))
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
