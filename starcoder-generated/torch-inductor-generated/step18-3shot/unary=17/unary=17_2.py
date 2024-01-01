
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = x1.permute(0, 3, 2, 1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
