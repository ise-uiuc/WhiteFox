
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch requires a module to initialize the convolution layer
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.conv = Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
