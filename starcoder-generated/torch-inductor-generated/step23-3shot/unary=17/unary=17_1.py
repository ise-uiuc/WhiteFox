
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 3, 5, stride=1, padding=3)
        self.conv2 = torch.nn.ConvTranspose2d(3, 1, 8, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return v2, v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
