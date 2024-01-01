
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(18, 15, (3, 9), stride=(1, 5), padding=(1, 0))
        self.conv2 = torch.nn.Conv2d(15, 12, (3, 8), stride=(1, 3), padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 18, 48, 48)
