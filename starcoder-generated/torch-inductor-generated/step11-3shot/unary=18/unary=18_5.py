
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, 1, 0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2.reshape((v2.shape[0], 16, -1))
        v4 = v3.permute(0, 2, 1)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
