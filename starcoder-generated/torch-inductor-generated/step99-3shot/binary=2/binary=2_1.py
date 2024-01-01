
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 512, 1, stride=1, padding=0)
    def forward(self, x):
        v = self.conv(x)
        r = torch.relu(v)
        return r
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
