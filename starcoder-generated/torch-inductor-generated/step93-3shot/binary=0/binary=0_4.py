
class Model(nn.Module):
    def __init__(self, kernel_sz, stride):
        super().__init__()
        self.conv = nn.Conv2d(2, 6, kernel_sz, stride=stride)

    def forward(self, x, padding=4):
        v = self.conv(x)
        v2 = v + padding
        return v2
# Inputs to the model
x = torch.randn(1, 2, 128, 128)
