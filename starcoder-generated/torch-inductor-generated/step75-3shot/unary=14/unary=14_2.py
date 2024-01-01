
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2_1 = torch.nn.ConvTranspose2d(150, 300, 2, stride=1, padding=1, bias=False)
        self.conv1_2 = torch.nn.Conv2d(64, 184, 1, stride=1, padding=0, bias=False)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(270, 315, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose2_1(x1)
        v2 = self.conv1_2(x1)
        v3 = v1 + v2
        v4 = self.conv_transpose1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6     
# Inputs to the model
x1 = torch.randn(1, 150, 64, 64)
