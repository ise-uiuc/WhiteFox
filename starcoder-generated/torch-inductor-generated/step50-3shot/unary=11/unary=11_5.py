
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(32, 64, 1)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 19
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64, 64)
