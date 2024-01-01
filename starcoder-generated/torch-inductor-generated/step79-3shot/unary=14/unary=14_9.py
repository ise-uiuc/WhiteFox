
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_9 = torch.nn.Conv2d(1, 384, (3, 3), stride=1, padding=(1, 1))
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(1, 384, 3, stride=1, padding=1, groups=2)
    def forward(self, x0):
        v0 = torch.sigmoid(x0)
        v1 = self.conv1_9(v0)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_9(v3)
        return v4
# Inputs to the model
x0 = torch.randn(1, 1, 10, 10)
