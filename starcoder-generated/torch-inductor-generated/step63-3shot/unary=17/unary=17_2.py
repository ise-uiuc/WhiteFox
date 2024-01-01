
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 64, 3, padding=2, stride=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 35, 35)
