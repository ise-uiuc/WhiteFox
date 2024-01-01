
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (4, 4), stride=(2, 2), padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, (4, 4), stride=(1, 1), padding=1)
        self.conv4 = torch.nn.Conv2d(16, 8, (4, 4), stride=(1, 1), padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(2, 1, 34, 84)
