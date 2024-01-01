
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_transpose5 = torch.nn.ConvTranspose2d(35, 70, 4, padding=(1, 1), stride=3, groups=5)
    def forward(self, x13):
        x1 = self.conv2d_transpose5(x13)
        return x1.relu()
# Inputs to the model
x13 = torch.randn(1, 35, 15, 21)
