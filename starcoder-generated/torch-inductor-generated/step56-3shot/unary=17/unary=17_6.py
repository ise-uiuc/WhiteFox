
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, padding=2, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, padding=2, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.sigmoid(v3)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
