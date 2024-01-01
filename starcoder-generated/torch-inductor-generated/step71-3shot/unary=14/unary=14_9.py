
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2_1 = torch.nn.Conv2d(3, 5, 3, stride=2, padding=2)
        self.conv_transpose_1_3 = torch.nn.ConvTranspose2d(5, 5, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv2_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1_3(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
