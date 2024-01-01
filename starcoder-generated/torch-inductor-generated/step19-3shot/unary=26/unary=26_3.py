
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(480, 480, 2, stride=2)
        self.linear = torch.nn.Linear(480*480, 7)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2.view(16, 480*480)
        x4 = self.linear(x3)
        return x4
# Inputs to the model
x1 = torch.randn(16, 480, 10, 10, 10)
