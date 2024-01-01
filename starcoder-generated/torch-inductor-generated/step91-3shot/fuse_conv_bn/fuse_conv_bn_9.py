
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=(4, 8), dilation=(2, 3))
    def forward(self, x):

        return self.conv1(x)
# Inputs to the model
x = torch.randn(1, 64, 56, 56)
