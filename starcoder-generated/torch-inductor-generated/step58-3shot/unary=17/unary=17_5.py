
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 2, stride=2, padding=2)
        self.conv = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(10, 5, 3, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.cat((v1, x2), 1)
        v3 = self.conv(v2)
        v4 = torch.relu(v3)
        v5 = self.conv1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
x2 = torch.randn(1, 3, 6, 6)
