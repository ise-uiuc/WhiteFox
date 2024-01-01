
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (1, 1), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 1, (1, 1), stride=1, padding=0)
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.conv2(v1)
        return v2
## Inputs to the model
x4 = torch.randn(1, 1, 1, 1)
