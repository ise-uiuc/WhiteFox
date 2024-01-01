
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, (7, 7), stride=(1, 1), padding=(3, 3))
    def forward(self, x):
        v = self.conv1(x)
        v = self.conv2(v)
        return v
# Inputs to the model
x = torch.randn(1, 8, 8, 8)
