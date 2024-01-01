
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=(2, 2), padding=1)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 - 0.25
        return v2
# Inputs to the model
x5 = torch.randn(1, 1, 20, 30)
