
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv_relu = torch.nn.Conv2d(3, 64, 7, stride=1, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_bn(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
