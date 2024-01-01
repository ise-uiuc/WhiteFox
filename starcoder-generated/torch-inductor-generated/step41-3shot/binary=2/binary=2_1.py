
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 56, 1, stride=2, padding=0)
        self.linear = torch.nn.Linear(56, 10)
    def forward(self, x):
        y = x.view(-1, 512*8*8)
        v1 = self.conv(y)
        v2 = self.linear(v1.view(-1, 56))
        v3 = v2 - 5
        return v3
# Inputs to the model
x = torch.randn(1, 512, 4, 4)
