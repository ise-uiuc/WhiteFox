
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (1, 1,), stride=(1, 1,), bias=False)
    def forward(self, x1):
        v1 = torch.add(x1, 0.45804857617378235)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
