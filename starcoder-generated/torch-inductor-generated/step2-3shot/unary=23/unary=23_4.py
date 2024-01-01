
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(6, 9, 1, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv_transpose1(x1)
        t2 = torch.atan(t1)
        t3 = self.conv_transpose2(t2)
        v1 = torch.atan(t3)

        return v1
# Inputs to the model
x1 = torch.randn(1, 6, 32, 64, 64)
