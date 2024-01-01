
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 8), stride=(3, 1), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(16, 16, (5, 9), stride=(5, 1), padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
