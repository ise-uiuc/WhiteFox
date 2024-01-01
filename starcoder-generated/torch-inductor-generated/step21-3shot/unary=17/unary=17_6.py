
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(2, 4, 3, stride=2, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
