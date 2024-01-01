
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 100, 6)
        self.conv2 = torch.nn.Conv2d(100, 1, 2, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 1, 1, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(self.conv2(v1))
        v3 = torch.log_softmax(self.conv3(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 256, 64)
