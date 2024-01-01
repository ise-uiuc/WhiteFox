
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(12, 7, 3, stride=1, padding=0)
        self.conv_transpose_1_2 = torch.nn.ConvTranspose2d(7, 7, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv_transpose_1_2(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 12, 224, 224)
