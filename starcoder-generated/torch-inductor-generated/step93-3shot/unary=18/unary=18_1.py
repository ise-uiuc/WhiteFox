
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 48, (7, 1), stride=(1, 2), padding=(3, 0))
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(48, 16, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(16, 6, (1, 7), stride=(1, 1), padding=(0, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = self.relu(v1)
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
