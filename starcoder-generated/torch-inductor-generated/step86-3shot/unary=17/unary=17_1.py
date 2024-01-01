
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(128, 16, 3, padding=1, stride=2, output_padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(16, 2, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 224, 224)
