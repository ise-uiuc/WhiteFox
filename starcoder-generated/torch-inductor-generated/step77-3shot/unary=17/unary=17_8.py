
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 5, stride=2, padding=2, output_padding=1, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 1, stride=1, padding=0, output_padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1.
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 + 1.
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
        v8 = v7 + 1.
        v9 = torch.relu(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 600, 600)
