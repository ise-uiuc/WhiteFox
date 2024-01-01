
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(16, 16, 7, stride=1, padding=3, output_padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
