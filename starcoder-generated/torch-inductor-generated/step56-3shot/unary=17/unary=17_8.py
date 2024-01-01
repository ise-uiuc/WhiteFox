
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 2, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.tanh(v4)
        v6 = torch.squeeze(v5, dim=0)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
