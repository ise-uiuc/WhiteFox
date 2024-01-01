
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.selu(v1)
        return F.sigmoid(v2)
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
