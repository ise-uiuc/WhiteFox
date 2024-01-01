
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_1 = torch.nn.MaxPool2d(2, stride=1)
        self.conv_1 = torch.nn.Conv2d(50, 100, 2, stride=1, padding=0)
        self.maxpool_3 = torch.nn.MaxPool2d(2, stride=2)
        self.convtranspose_4 = torch.nn.ConvTranspose2d(5, 10, 4, stride=4, padding=0)
        self.conv_3 = torch.nn.Conv2d(10, 50, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.maxpool_1(x1)
        v2 = self.conv_1(v1)
        v3 = self.maxpool_3(v2)
        v4 = self.convtranspose_4(v3)
        v5 = self.conv_3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 50, 4451, 3293)
