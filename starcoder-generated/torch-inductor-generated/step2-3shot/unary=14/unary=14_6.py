
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.conv_transpose2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
