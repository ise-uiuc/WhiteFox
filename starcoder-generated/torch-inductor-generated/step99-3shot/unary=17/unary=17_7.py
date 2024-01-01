
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 5, stride=2, padding=1, output_padding=0, groups=1, bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 5, stride=2, padding=1, output_padding=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return torch.cat([v4, v4], dim=1)
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
