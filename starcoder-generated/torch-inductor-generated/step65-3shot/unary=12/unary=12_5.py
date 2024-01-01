
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (16,16), stride=(1, 1), padding=(7,7))
        self.conv_next = torch.nn.Conv2d(1, 1, 1, 1, 0)
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v1 = v1.add(x1)
        v1 = self.conv1(v1)
        v2 = torch.sigmoid(v1)
        v2 = v1.mul(v2)
        v3 = self.conv_next(v2)
        v4 = torch.sigmoid(v3)
        v4 = v3.mul(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
