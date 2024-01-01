
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
