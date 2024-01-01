
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 16, 3, stride=2, output_padding=1, padding=1, dilation=1, groups=1)
        self.pool = torch.nn.ReLU(0.140625)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.flatten(v1)
        v3 = torch.tanh(v2)
        v4 = v3.reshape(torch.Size([-1]))
        v5 = torch.relu(v4)
        v6 = self.pool(v5)
        return torch.tanh(v6)
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
