
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 6, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=2, dilation=2)
        self.conv3 = torch.nn.ConvTranspose2d(3, 32, 4, stride=2, padding=2, dilation=2)
        self.conv4 = torch.nn.ConvTranspose1d(32, 1, 2, stride=2, padding=1, dilation=10)
        self.conv5 = torch.nn.Conv1d(1, 7, 2, stride=1, padding=10, dilation=10)
    def forward(self, x):
        v1 = torch.sigmoid(self.conv1(x))
        v2 = torch.tanh(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = torch.relu(self.conv5(v4))
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 14, 100, 100)
