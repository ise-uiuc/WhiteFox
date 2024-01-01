
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_conv = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)
    def forward(self, x1):
        v1 = self.depth_conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
