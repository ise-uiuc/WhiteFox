
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose1 = torch.nn.ConvTranspose2d(9, 2, 5, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.convtranspose1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv2(v6)
# Inputs to the model
x1 = torch.randn(1, 9, 9, 9)
