
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, 3, stride=1,  padding=0)
        self.conv1 = self.conv
        self.conv1_transpose = self.conv_transpose
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1_transpose(v1)
        v3 = torch.relu(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
