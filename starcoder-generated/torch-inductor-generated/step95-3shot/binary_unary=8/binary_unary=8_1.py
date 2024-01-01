
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(32, 64, (6, 3), stride=(1, 1), padding=(1, 0), dilation=(3, 4), groups=32, bias=True)
    def forward(self, x):
        return torch.add(self.c1(x), x)
# Inputs to the model
x1 = torch.randn(1, 32, 113, 73)
x2 = torch.randn(1, 32, 113, 73)
