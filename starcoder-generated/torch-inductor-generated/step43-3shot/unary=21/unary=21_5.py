
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, 2)
        self.conv2 = torch.nn.Conv2d(128, 128, 2)
        self.conv3 = torch.nn.Conv2d(128, 128, 2)
        self.conv4 = torch.nn.Conv2d(128, 2, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        return torch.tanh(v7)
# Inputs to the model
x = torch.randn(10, 128, 128, 128)
