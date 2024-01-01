
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(8, 4, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(8, 64, 8, 8)
