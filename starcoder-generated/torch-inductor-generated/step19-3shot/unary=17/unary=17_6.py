
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 32, 1) # conv1
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 16, 2) # conv2_transpose
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
