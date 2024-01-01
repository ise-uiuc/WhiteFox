
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 3, 3, padding=1, stride=2)
        self.conv1 = torch.nn.Conv2d(2, 6, 3, padding=1, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(6, 1, 7, padding=3, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
