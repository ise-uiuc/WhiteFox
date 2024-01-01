
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Block begins
        self.conv = torch.nn.Conv2d(3, 8, 3, input_padding=(1, 1), stride=(2, 2))
        self.bn = torch.nn.BatchNorm2d(8, momentum=0.95, eps=0.2)
        # Block ends
        # Note: add here after block 1 (optional)
    def forward(self, x1):
        # Output tensor for block 1
        y1 = None
        # Block begins
        v1 = self.conv(x1)
        v1 = self.bn(v1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.sin(v3)
        v5 = torch.cosh(v4)
        y1 = v5
        # Block ends
        return y1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
