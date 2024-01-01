
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
