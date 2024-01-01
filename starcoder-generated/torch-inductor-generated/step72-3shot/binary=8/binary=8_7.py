
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(300, 500, 7, stride=1, padding=7 // 2 + 1, groups=1)
    def forward(self, x1):
        return self.conv1(x1)
# Input to the model
x1 = torch.randn(1, 300, 80, 80)
