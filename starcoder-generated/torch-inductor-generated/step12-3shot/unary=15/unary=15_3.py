
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        # Should be relu(conv(relu(conv(bn)))) but relu() is not supported, and cannot convert it into v4 = relu(conv_bn)
        # Also conv() is not supported, therefore cannot convert the last conv_bn into v4 = relu(conv(bn))
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
