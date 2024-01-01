
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.max1 = nn.MaxPool1d(8, 1, 7)
        self.max2 = nn.MaxPool1d(10, 2, 5)
    def forward(self, x):
        x = self.max1(x)
        x = self.max2(x)
        x = x.flatten(1)
        x = x * 0.0
        return x
# Inputs to the model
x = torch.randn(2, 1, 10)
