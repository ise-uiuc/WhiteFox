
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(2, 1), groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.Sigmoid()(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
