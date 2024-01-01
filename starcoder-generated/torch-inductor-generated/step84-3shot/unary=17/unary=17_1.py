
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 16, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.max_pool2d(v2, 2, stride=2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
