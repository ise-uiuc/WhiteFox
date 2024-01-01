
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 32, 5, stride=1, padding=2, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 128, 3, padding=1, stride=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = torch.relu(v1)
        v4 = self.conv_transpose(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 64, 128, 128)
