
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        s = self.relu(x)
        t = self.relu(x)
        y = s + t
        return y
# Inputs to the model
x = torch.randn(1, 3, 2, 2)
