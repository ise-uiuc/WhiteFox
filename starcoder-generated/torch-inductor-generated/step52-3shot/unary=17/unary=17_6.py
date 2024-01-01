
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 5, stride=2, padding=2, output_padding=0, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 1, stride=1, padding=0, output_padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
