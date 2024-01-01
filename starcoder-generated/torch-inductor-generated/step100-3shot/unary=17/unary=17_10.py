
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(4, 8, kernel_size=3, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv_transpose2d(x1)
        v2 = x1[:, :, 0:4]
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 35, 35)
x2 = torch.randn(1, 8, 35, 35)
