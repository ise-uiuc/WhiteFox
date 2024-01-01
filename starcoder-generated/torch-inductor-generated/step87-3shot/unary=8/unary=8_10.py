
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 2, 3, stride=1, padding=0, output_padding=0)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 3, 5, stride=2, padding=1, output_padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(3, 4, 5, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v1 = self.bn1(v1)
        v2 = self.conv_transpose2(v1)
        v2 = self.bn2(v2)
        v3 = self.conv_transpose3(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v4 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
