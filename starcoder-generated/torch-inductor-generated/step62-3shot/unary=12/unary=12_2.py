
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 2, 1)
        self.conv2 = torch.nn.Conv2d(2, 2, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v2)
        v4 = torch.mul(v2, v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 2, 30)
