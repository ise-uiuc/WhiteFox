
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 4, 2, 1)
    def forward(self, x):
        m2 = torch.tanh(self.conv(x))
        return m2
# Inputs to the model
x = torch.rand(1, 3, 16, 16)
