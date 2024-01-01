
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.softmax1 = torch.nn.Softmax2d()
        self.conv2 = torch.nn.Conv2d(3, 7, 5, stride=(2, 1), padding=0)
        self.softmax2 = torch.nn.Softmax2d()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.softmax1(v1)
        v3 = self.conv2(v2)
        v4 = self.softmax2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
