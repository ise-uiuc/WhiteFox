
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, (5, 2), stride=(5, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(2, 8, (5, 3), stride=(5, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(8, 32, (5, 2), stride=(5, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
