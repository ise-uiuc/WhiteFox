
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = torch.nn.BatchNorm1d(63)
        self.conv1 = torch.nn.Conv2d(58, 79, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.b1(x1)
        v2 = self.conv1(v1)
        if other == None:
            other = torch.randn(v2.shape).to(x1.device)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 58, 64, 64).to('cpu')
