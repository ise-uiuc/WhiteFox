
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
