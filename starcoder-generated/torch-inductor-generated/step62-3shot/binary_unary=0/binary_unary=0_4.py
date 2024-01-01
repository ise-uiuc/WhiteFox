
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.linear = torch.nn.Linear(512, 512)
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), groups=256, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.linear(v1)
        v3 = self.conv2(v2)
        v4 = self.conv1(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)
x2 = torch.randn(46, 512)
