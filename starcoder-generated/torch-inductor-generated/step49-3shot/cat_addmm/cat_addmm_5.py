
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)
        self.conv = nn.Conv2d(12, 24, 3, stride=2, padding=1)
    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x.reshape(x.shape[0], 4, -1))
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 9)
