
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, np.random.randint(1, 5), stride=2, padding=6)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 350
        return v2
# Inputs to the model
x = torch.randn(1, 3, 8, 8)
