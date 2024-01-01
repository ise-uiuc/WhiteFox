
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, 5, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 10, 5, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 - -10.22
        v5 = torch.nn.functional.relu(v4)
        v6 = v5 - 0.9
        v7 = torch.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
