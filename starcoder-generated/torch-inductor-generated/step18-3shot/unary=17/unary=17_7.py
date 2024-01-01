
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(3, 3, 3, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 16, 16, 16)
