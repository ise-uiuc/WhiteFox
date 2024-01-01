
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(331, 3, (20, 20), stride=(1, 10), padding=(3, 5))
        self.conv2 = torch.nn.Conv2d(3, 1, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(1, 3, (3, 3), stride=(4, 4), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 331, 60, 10)
