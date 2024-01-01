
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, 7, stride=3, padding=3, dilation=1)
        print(self.conv_1.weight.shape)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
