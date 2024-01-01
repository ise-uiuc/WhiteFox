
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(56, 47, 1, stride=1, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(47, 15, 1, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(15, 26, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv_transpose1(v1)
        v3 = self.conv_transpose2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 56, 6, 6)
