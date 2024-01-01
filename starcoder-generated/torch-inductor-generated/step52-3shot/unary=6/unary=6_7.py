
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, (2, 2), stride=(2, 2), padding=(1, 1))
        self.prelu = torch.nn.PReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = torch.nn.ELU()
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = self.prelu(x1)
        x3 = self.max_pool(x2)
        x4 = self.act(x3)
        x5 = self.bn(x4)
        return x5
# Inputs to the model
x1 = torch.randn(5, 3, 224, 224)
