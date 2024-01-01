
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 9, kernel_size=(2, 1), stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 14, kernel_size=(3, 1), stride=2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(14, 18, kernel_size=(2, 2), stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = torch.clamp(v5, max=6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
