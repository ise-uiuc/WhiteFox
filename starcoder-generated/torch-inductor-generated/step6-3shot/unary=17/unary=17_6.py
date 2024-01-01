
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(3, 32, 3, padding=1, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose3d(1, 16, 3, padding=4, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv_transpose0(x1)
        v2 = F.relu(v1)
        v3 = torch.rand_like(x2)
        v4 = (x2 + v3)
        v5 = self.conv_transpose1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn((1, 1, 16, 16, 16))
