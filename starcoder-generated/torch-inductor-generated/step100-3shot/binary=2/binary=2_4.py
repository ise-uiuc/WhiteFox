
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, (3, 2), stride=(1, 2), padding=(2, 1))
        self.conv1 = torch.nn.Conv2d(7, 8, (3, 2), stride=(3, 2), padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1(v1)
        v3 = v2 - 0.1
        return v3
# Inputs to the model
x = torch.randn(1, 3, 32, 64)
