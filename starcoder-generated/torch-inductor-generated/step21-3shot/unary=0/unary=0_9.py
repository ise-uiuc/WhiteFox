
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 100, 5, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(100, 200, 3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(200, 300, 2, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 416, 416)
