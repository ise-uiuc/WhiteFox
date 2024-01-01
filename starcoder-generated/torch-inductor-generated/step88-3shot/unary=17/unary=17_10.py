
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(128, 128, 3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = torch.sigmoid(v1)
        v4 = self.conv(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 128, 128)
