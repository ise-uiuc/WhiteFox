
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 1, 7, stride=1, padding=3)
        self.conv1 = torch.nn.ConvTranspose2d(1, 4, 3, stride=3, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 1, 1, bias=True, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(x1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
