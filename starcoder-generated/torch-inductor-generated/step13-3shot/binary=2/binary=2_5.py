
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, (1, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = [0.16, 0.90, 0.78, 0.22] - v1
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
