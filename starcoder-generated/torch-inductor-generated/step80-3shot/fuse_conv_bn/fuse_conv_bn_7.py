
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 2, 1)
        self.linear = torch.nn.Linear(3, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        x = self.conv(x)
        x = x.detach()
        x = self.linear(x)
        return self.softmax(x)
# Inputs to the model
x = torch.randn(1, 3, 8, 8)
