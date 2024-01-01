
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(42, 64, 8, stride=4, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 64, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 42, 6, 7)
