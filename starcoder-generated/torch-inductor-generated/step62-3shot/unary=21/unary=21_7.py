
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 3, 1, stride=1, padding=0)
    def forward(self, x):
        ret1 = torch.tanh(self.conv1(x))
        return ret1
# Inputs to the model
x = torch.randn(1, 512, 17, 260)
