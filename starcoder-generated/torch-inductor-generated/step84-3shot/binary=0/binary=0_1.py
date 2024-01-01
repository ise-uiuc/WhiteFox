
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 22, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(41472, 4096)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = torch.sum(v2, dim=1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64).to('cpu')
