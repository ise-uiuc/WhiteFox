
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = torch.nn.functional.pad(v1, (0, 0, 0, 0), 'constant', 8414204576.0)
        v4 = 71617988457.0
        return v3
# Inputs to the model
x1 = torch.randn(4, 1, 28, 28)
