
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.max_pool2d(v3, 2, stride=1)
        v5 = self.conv1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)
