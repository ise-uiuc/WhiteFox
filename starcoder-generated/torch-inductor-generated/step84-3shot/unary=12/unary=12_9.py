
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, kernel_size=(1, 6), stride=(1, 2), padding=(0, 1), groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 3, 5)
