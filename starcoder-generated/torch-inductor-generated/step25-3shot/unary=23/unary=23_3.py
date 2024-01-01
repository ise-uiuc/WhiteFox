
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(1, 10, 2, stride=2, padding=2, output_padding=1) 
        self.conv = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(4)
    def forward(self, x1):
        v1 = self.tconv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        v5 = self.avgpool(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
