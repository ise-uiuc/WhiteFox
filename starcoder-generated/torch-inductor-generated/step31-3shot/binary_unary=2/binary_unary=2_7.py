
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, 5, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 10, 5, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - -10.22
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 0.9
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
