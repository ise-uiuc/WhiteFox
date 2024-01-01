
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(4, 2, 3, stride=2)
        self.flatten = torch.flatten
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.flatten(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
