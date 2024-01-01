
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 8, 3)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(12, 24, 5, stride=3, padding=2)
        self.conv3 = torch.nn.Conv2d(18, 6, 7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = self.conv3(v1)
        v2 = self.conv_transpose2(v3)
        v4 = torch.sigmoid(v2)
        v5 = v2 * v4
        return v5, v1, v3, v2
# Inputs to the model
x1 = torch.randn(1, 7, 224, 224)
