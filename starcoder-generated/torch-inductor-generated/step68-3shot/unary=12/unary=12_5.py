
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m1 = torch.nn.Conv2d(3, 8, 3, stride=(2, 1), padding=1, dilation=1)
        self.conv_1 = torch.nn.Conv2d(3, 8, 3, stride=(1, 2), padding=1, dilation=1)
        self.conv_2 = torch.nn.Conv2d(3, 8, 3, stride=(1, 2), padding=2, dilation=2)
        self.conv_3 = torch.nn.Conv2d(3, 8, 3, stride=(1, 2), padding=5, dilation=5)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(x1)
        v3 = self.conv_3(x1)
        v4 = torch.sigmoid(v1 + v2 + v3)
        v5 = v1*v4+v2*v4+v3*v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
