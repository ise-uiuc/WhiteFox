
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7)
        self.pool1 = torch.nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.pool1(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
