
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        return torch.tanh(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
