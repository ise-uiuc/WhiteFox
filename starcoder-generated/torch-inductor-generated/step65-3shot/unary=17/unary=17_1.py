
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 3, 7, stride=1, padding=3)
        self.conv1 = torch.nn.ConvTranspose2d(3, 2, 11, stride=1, padding=5, output_padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        return torch.squeeze(v6, dim=0)
# Inputs to the model
x1 = torch.randn(1, 1, 51, 51)
