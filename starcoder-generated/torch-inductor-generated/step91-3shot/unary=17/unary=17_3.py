
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 16, 3, stride=1)
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv(v2)
        v5 = torch.sigmoid(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 24, 24)
