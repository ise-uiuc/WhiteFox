
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 3, stride=2, padding=10)
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x_train = torch.randn(3, 1, 16, 16)
