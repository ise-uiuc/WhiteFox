
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(128, 1, 1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
