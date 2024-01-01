
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 9, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 3.89
        v4 = torch.tanh(v3)
        v5 = torch.relu(v4)
        v6 = v5[:, 0, :, :]
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
