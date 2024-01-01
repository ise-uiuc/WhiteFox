
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 6, 3, stride=2, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(6, 8, 3, stride=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
