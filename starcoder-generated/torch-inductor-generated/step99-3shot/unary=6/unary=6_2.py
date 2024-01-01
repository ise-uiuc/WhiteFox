
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        x1 = self.avgpool(x1)
        x1 = self.conv(x1)
        return self.tanh(x1)
# Inputs to the model
x1 = torch.randn(1,3,1,1)
