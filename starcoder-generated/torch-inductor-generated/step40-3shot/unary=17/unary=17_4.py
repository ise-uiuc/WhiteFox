
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = v3.transpose(3, 2)
        v5 = v4.transpose(3, 2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
