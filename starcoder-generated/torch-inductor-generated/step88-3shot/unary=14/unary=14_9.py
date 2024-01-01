
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(64, 3, 1, stride=1, padding=0, bias=False)
        self.conv_transpose1_2 = torch.nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_transpose1_2(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 124, 28)
